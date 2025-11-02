import json

with open('/home/shaofengyin/AgenticVerifier/output/static_scene/20251028_133713/christmas1/generator_memory.json', 'r') as f:
    memory = json.load(f)
    
count = 1

for message in memory:
    if message['role'] == 'assistant':
        for tool_call in message['tool_calls']:
            if tool_call['function']['name'] == 'execute_and_evaluate':
                tool_call['function']['arguments'] = json.loads(tool_call['function']['arguments'])
                code = tool_call['function']['arguments']['full_code']
                with open(f'/home/shaofengyin/AgenticVerifier/output/static_scene/20251028_133713/christmas1/scripts/{count}.py', 'w') as f:
                    f.write(code)
                count += 1
                if count > 11:
                    break