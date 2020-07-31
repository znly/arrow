// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! Contains writer which writes arrow data into parquet data.

use std::fs::File;
use std::rc::Rc;

use array::Array;
use arrow::array;
use arrow::datatypes::{DataType as ArrowDataType, Field, Schema};
use arrow::record_batch::RecordBatch;

use crate::column::writer::ColumnWriter;
use crate::errors::{ParquetError, Result};
use crate::file::properties::WriterProperties;
use crate::{
    data_type::*,
    file::writer::{FileWriter, RowGroupWriter, SerializedFileWriter},
    schema::types::ColumnDescPtr,
};

pub struct ArrowWriter {
    writer: SerializedFileWriter<File>,
    columns: Vec<DataTypeWithDefAndRepLevel>,
    total_num_rows: i64,
}
#[derive(Debug)]
struct DataTypeWithDefAndRepLevel {
    data_type: ArrowDataType,
    def_levels: Option<Vec<i16>>,
    rep_levels: Option<Vec<i16>>,
}

impl ArrowWriter {
    pub fn try_new(file: File, arrow_schema: &Schema) -> Result<Self> {
        let schema = crate::arrow::arrow_to_parquet_schema(arrow_schema)?;
        let props = Rc::new(WriterProperties::builder().build());
        let file_writer = SerializedFileWriter::new(
            file.try_clone()?,
            schema.root_schema_ptr(),
            props,
        )?;

        Ok(Self {
            writer: file_writer,
            columns: flattened_column_types(
                arrow_schema.fields(),
                Vec::new().as_mut(),
                Vec::new().as_mut(),
            )?,
            total_num_rows: 0,
        })
    }

    pub fn write(&mut self, batch: &RecordBatch) -> Result<()> {
        let mut row_group_writer = self.writer.next_row_group()?;

        for (column_descriptor, column) in self.columns.iter().zip(batch.columns()) {
            let mut writer = row_group_writer
                .next_column()?
                .ok_or_else(|| ParquetError::General("No writer found".to_string()))?;
            self.total_num_rows += write_column(
                &mut writer,
                column,
                column_descriptor
                    .def_levels
                    .as_ref()
                    .map(|lvls| lvls.as_slice()),
                column_descriptor
                    .rep_levels
                    .as_ref()
                    .map(|lvls| lvls.as_slice()),
            )? as i64;
            row_group_writer.close_column(writer)?;
        }
        self.writer.close_row_group(row_group_writer)
    }

    pub fn close(&mut self) -> Result<()> {
        self.writer.close()
    }
}

fn flattened_column_types(
    fields: &Vec<Field>,
    current_def_levels: &mut Vec<i16>,
    current_rep_levels: &mut Vec<i16>,
) -> Result<Vec<DataTypeWithDefAndRepLevel>> {
    let mut column_types = Vec::new();
    for field in fields.iter() {
        match field.data_type() {
            ArrowDataType::Struct(fields) => {
                current_rep_levels.push(1);
                column_types.append(&mut flattened_column_types(
                    fields,
                    current_def_levels,
                    current_rep_levels,
                )?);
            }
            ArrowDataType::List(_dtype) => unimplemented!("list not yet implemented"),
            ArrowDataType::LargeBinary => {
                unimplemented!("large binary not yet implemented")
            }
            ArrowDataType::LargeUtf8 => unimplemented!("large utf8 not yet implemented"),
            ArrowDataType::LargeList(_) => {
                unimplemented!("large lilst not yet implemented")
            }
            ArrowDataType::FixedSizeList(_, _) => {
                unimplemented!("fsl not yet implemented")
            }
            ArrowDataType::Null => unimplemented!(),
            ArrowDataType::Union(_) => unimplemented!(),
            ArrowDataType::Dictionary(_, _) => unimplemented!(),
            data_type => {
                let mut def_levels = if current_def_levels.len() > 0 {
                    current_def_levels.clone()
                } else {
                    Vec::new()
                };
                if field.is_nullable() {
                    def_levels.push(def_levels.last().unwrap_or(&0) + 1);
                } else {
                    def_levels.push(1);
                }

                let rep_levels = if current_rep_levels.len() > 0 {
                    Some(current_rep_levels.clone())
                } else {
                    None
                };
                // TODO [igni]: remove the clone ?
                column_types.push(DataTypeWithDefAndRepLevel {
                    data_type: data_type.clone(),
                    def_levels: def_levels.into(),
                    rep_levels,
                });
            }
        }
    }
    Ok(column_types)
}

/// Write column to writer
fn write_column(
    writer: &mut ColumnWriter,
    column: &array::ArrayRef,
    def_levels: Option<&[i16]>,
    rep_levels: Option<&[i16]>,
) -> Result<usize> {
    match writer {
        ColumnWriter::Int32ColumnWriter(ref mut typed) => {
            let array = array::Int32Array::from(column.data());
            typed.write_batch(
                get_numeric_array_slice::<Int32Type, _>(&array).as_slice(),
                def_levels,
                rep_levels,
            )
        }
        ColumnWriter::BoolColumnWriter(ref mut _typed) => unimplemented!(),
        ColumnWriter::Int64ColumnWriter(ref mut typed) => {
            let array = array::Int64Array::from(column.data());
            typed.write_batch(
                get_numeric_array_slice::<Int64Type, _>(&array).as_slice(),
                def_levels,
                rep_levels,
            )
        }
        ColumnWriter::Int96ColumnWriter(ref mut _typed) => unimplemented!(),
        ColumnWriter::FloatColumnWriter(ref mut typed) => {
            let array = array::Float32Array::from(column.data());
            typed.write_batch(
                get_numeric_array_slice::<FloatType, _>(&array).as_slice(),
                def_levels,
                rep_levels,
            )
        }
        ColumnWriter::DoubleColumnWriter(ref mut typed) => {
            let array = array::Float64Array::from(column.data());
            typed.write_batch(
                get_numeric_array_slice::<DoubleType, _>(&array).as_slice(),
                def_levels,
                rep_levels,
            )
        }
        ColumnWriter::ByteArrayColumnWriter(ref mut typed) => {
            let array = array::BinaryArray::from(column.data());

            let mut values: Vec<ByteArray> =
                Vec::with_capacity(array.len() - array.null_count());
            for i in 0..array.len() {
                if array.is_valid(i) {
                    values.push(ByteArray::from(array.value(i).to_vec()))
                }
            }
            typed.write_batch(values.as_slice(), def_levels, rep_levels)
        }
        ColumnWriter::FixedLenByteArrayColumnWriter(ref mut _typed) => unimplemented!(),
    }
}

/// Get the definition levels of the numeric array, with level 0 being null and 1 being not null
/// In the case where the array in question is a child of either a list or struct, the levels
/// are incremented in accordance with the `level` parameter.
/// Parent levels are either 0 or 1, and are used to higher (correct terminology?) leaves as null
fn get_primitive_def_levels(
    array: &array::ArrayRef,
    level: i16,
    parent_levels: &[i16],
) -> Vec<i16> {
    // convince the compiler that bounds are fine
    let len = array.len();
    assert_eq!(
        len,
        parent_levels.len(),
        "Parent definition levels must equal array length"
    );
    let levels = (0..len)
        .map(|index| (array.is_valid(index) as i16 + level) * parent_levels[index])
        .collect();
    levels
}

/// Get the underlying numeric array slice, skipping any null values.
/// If there are no null values, the entire slice is returned,
/// thus this should only be called when there are null values.
fn get_numeric_array_slice<T, A>(array: &array::PrimitiveArray<A>) -> Vec<T::T>
where
    T: DataType,
    A: arrow::datatypes::ArrowNumericType,
    T::T: From<A::Native>,
{
    let mut values = Vec::with_capacity(array.len() - array.null_count());
    for i in 0..array.len() {
        if array.is_valid(i) {
            values.push(array.value(i).into())
        }
    }
    values
}

/// Get the underlying numeric array slice, skipping any null values.
/// If there are no null values, the entire slice is returned,
/// thus this should only be called when there are null values.
fn get_list_array_slice<T, A>(array: &array::PrimitiveArray<A>) -> Vec<T::T>
where
    T: DataType,
    A: arrow::datatypes::ArrowNumericType,
    T::T: From<A::Native>,
{
    let mut values = Vec::with_capacity(array.len() - array.null_count());
    for i in 0..array.len() {
        if array.is_valid(i) {
            values.push(array.value(i).into())
        }
    }
    values
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::sync::Arc;

    use arrow::array::*;
    use arrow::datatypes::ToByteSlice;
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;

    #[test]
    fn arrow_writer() {
        // define schema
        let schema = Schema::new(vec![
            Field::new("a", DataType::Int32, false),
            Field::new("b", DataType::Int32, true),
        ]);

        // create some data
        let a = Int32Array::from(vec![1, 2, 3, 4, 5]);
        let b = Int32Array::from(vec![Some(1), None, None, Some(4), Some(5)]);

        // build a record batch
        let batch = RecordBatch::try_new(
            Arc::new(schema.clone()),
            vec![Arc::new(a), Arc::new(b)],
        )
        .unwrap();

        let file = File::create("test.parquet").unwrap();
        let mut writer = ArrowWriter::try_new(file, &schema).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
    }

    #[test]
    fn arrow_writer_complex() {
        // define schema
        let struct_field_d = Field::new("d", DataType::Float64, true);
        let struct_field_f = Field::new("f", DataType::Float32, true);
        let struct_field_g =
            Field::new("g", DataType::List(Box::new(DataType::Boolean)), false);
        let struct_field_e = Field::new(
            "e",
            DataType::Struct(vec![
                struct_field_f.clone(),
                // struct_field_g.clone()
            ]),
            true,
        );
        let schema = Schema::new(vec![
            Field::new("a", DataType::Int32, false),
            Field::new("b", DataType::Int32, true),
            Field::new(
                "c",
                DataType::Struct(vec![struct_field_d.clone(), struct_field_e.clone()]),
                false,
            ),
        ]);

        // create some data
        let a = Int32Array::from(vec![1, 2, 3, 4, 5]);
        let b = Int32Array::from(vec![Some(1), None, None, Some(4), Some(5)]);
        let d = Float64Array::from(vec![None, None, None, Some(1.0), None]);
        let f = Float32Array::from(vec![Some(0.0), None, Some(333.3), None, Some(5.25)]);

        let g_value = BooleanArray::from(vec![
            false, true, false, true, false, true, false, true, false, true,
        ]);

        // Construct a buffer for value offsets, for the nested array:
        //  [[false], [true, false], null, [true, false, true], [false, true, false, true]]
        let g_value_offsets =
            arrow::buffer::Buffer::from(&[0, 1, 3, 3, 6, 10].to_byte_slice());

        // Construct a list array from the above two
        let g_list_data = ArrayData::builder(struct_field_g.data_type().clone())
            .len(5)
            .add_buffer(g_value_offsets.clone())
            .add_child_data(g_value.data())
            .build();
        let _g = ListArray::from(g_list_data);

        let e = StructArray::from(vec![
            (struct_field_f, Arc::new(f) as ArrayRef),
            // (struct_field_g, Arc::new(g) as ArrayRef),
        ]);

        let c = StructArray::from(vec![
            (struct_field_d, Arc::new(d) as ArrayRef),
            (struct_field_e, Arc::new(e) as ArrayRef),
        ]);

        // build a record batch
        let batch = RecordBatch::try_new(
            Arc::new(schema.clone()),
            vec![Arc::new(a), Arc::new(b), Arc::new(c)],
        )
        .unwrap();

        let file = File::create("test_complex.parquet").unwrap();
        let mut writer = ArrowWriter::try_new(file, &schema).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
    }
}
