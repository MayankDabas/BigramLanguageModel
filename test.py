import json

# Data to be written
config_data = {
    'name': 'John',
    'role': 'developer',
    'languages': ['Python', 'JavaScript']
}

# Specifying the file name
config_filename = 'config.json'

for i in range(2):
    # Writing the dictionary to a file in JSON format
    with open(config_filename, 'w') as config_file:
        json.dump(config_data, config_file)

print(f"Data successfully written to {config_filename}")

with open(config_filename, 'r') as f:
    data = json.load(f)
print(data['languages'])