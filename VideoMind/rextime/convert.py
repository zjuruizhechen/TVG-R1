import json

input_path = "rextime_val.jsonl"
output_path = "rextime_val.json"

# 读取 jsonl 文件中的每一行，解析为 JSON 对象，组成一个列表
with open(input_path, 'r', encoding='utf-8') as fin:
    data = [json.loads(line) for line in fin]

# 写入 json 文件（一个完整的 JSON 数组）
with open(output_path, 'w', encoding='utf-8') as fout:
    json.dump(data, fout, indent=2, ensure_ascii=False)
