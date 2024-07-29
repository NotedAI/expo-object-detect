import os

# List of shard files in order
shard_files = [
    'group1-shard1of5.bin',
    'group1-shard2of5.bin',
    'group1-shard3of5.bin',
    'group1-shard4of5.bin',
    'group1-shard5of5.bin',
]

# Output file
output_file = 'combined_output.bin'

# Open the output file in write-binary mode
with open(output_file, 'wb') as outfile:
    # Iterate over each shard file
    for shard in shard_files:
        # Open each shard file in read-binary mode
        with open(shard, 'rb') as infile:
            # Read the contents of the shard file and write it to the output file
            outfile.write(infile.read())

print(f"Shards have been combined into {output_file}")
