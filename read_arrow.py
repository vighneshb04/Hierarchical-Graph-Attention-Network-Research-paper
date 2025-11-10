import pyarrow as pa
import pyarrow.ipc as ipc

arrow_file = 'json/train/data-00000-of-00001.arrow'
with pa.memory_map(arrow_file, 'r') as source:
    reader = ipc.RecordBatchFileReader(source)
    table = reader.read_all()
    df = table.to_pandas()
    print(df.head())
