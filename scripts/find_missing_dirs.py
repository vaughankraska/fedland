# List of IDs from experiments.json
ids = [
    "b458a5f7-193a-46f4-97d5-aac510bb5952",
    "64b57f5d-dbaf-4ffb-b9db-396ca6f73edc",
    "4b3a8ef9-c2ee-4a01-9260-6e4af635dc03",
    "f36e2c39-34cb-4b03-a2f9-68cb33bdbebe",
    "8b1d3a21-7e63-40e7-be21-16d69b0b0aac",
    "735bb18d-fb33-4be3-adaf-78289b69319d",
    "df3a0e29-0fc7-4cd5-838f-326822e098fc",
    "3f6c270c-ff3d-4c1d-a01c-3359312cd3c4",
    "5f5b1857-84e7-4620-aa9c-77f9ea5e12da",
    "9d0054da-149f-4a8b-aa2c-6583ba8f7484",
    "d9ae0c7d-e2bb-4ee4-94e2-4870ae5eb80a",
    "c986f32b-b752-45d2-ae81-9ba9b48289f7",
    "01960312-13a0-4f41-9e14-31508c3c02ba",
    "b4e8e959-5146-4418-9a89-ffe9def5dcbd",
    "2bcc3999-3677-4989-a7eb-32639b83c165",
    "176b58e9-aace-4cbc-b7dc-8dba88af1bff",
    "386b3021-3e83-4f5a-805b-3cdfcadb2489",
    "ba41442b-e46e-41a1-870c-ff313b6a0c8c",
    "3991aac2-cc3c-4405-b26f-e2896394190a",
    "7d546014-9e46-4ada-b86a-dc8b15b0d478",
    "5230e970-1ee8-4d61-9fde-83624c02d771",
    "e6c4fbe6-b338-4ddb-9994-eb3097535278",
    "bb3aa1b6-db01-4610-9689-9d0101e92d17",
    "9a0fe8d3-8506-4157-ab61-291fc6d9c38d",
    "a97f0e29-0262-4eb9-a117-f08f25a3a2b1",
    "ef47fd93-925a-42f4-be44-7b302cee44d3",
    "7bf0af61-2674-4fec-960f-bbf6fe5c2d36"
]

# List of directories
directories = [
    "01960312-13a0-4f41-9e14-31508c3c02ba",
    "176b58e9-aace-4cbc-b7dc-8dba88af1bff",
    "2bcc3999-3677-4989-a7eb-32639b83c165",
    "31b1e29c-a099-4ce4-b8e4-46bb75b41d5e",
    "386b3021-3e83-4f5a-805b-3cdfcadb2489",
    "3991aac2-cc3c-4405-b26f-e2896394190a",
    "3f6c270c-ff3d-4c1d-a01c-3359312cd3c4",
    "410ad360-b445-4ae8-b224-6f997845a9b1",
    "4b3a8ef9-c2ee-4a01-9260-6e4af635dc03",
    "5230e970-1ee8-4d61-9fde-83624c02d771",
    "5f5b1857-84e7-4620-aa9c-77f9ea5e12da",
    "630c180e-e292-4fc3-8f7c-6f6655855a5b",
    "64b57f5d-dbaf-4ffb-b9db-396ca6f73edc",
    "735bb18d-fb33-4be3-adaf-78289b69319d",
    "7bf0af61-2674-4fec-960f-bbf6fe5c2d36",
    "7d546014-9e46-4ada-b86a-dc8b15b0d478",
    "8b1d3a21-7e63-40e7-be21-16d69b0b0aac",
    "9070ebff-43ab-4eeb-9a8a-d64024ee8e04",
    "9a0fe8d3-8506-4157-ab61-291fc6d9c38d",
    "9d0054da-149f-4a8b-aa2c-6583ba8f7484",
    "a124fab0-7aa4-4c09-ae01-6a89a619b812",
    "a97f0e29-0262-4eb9-a117-f08f25a3a2b1",
    "b451d047-c9b3-450c-be7b-42c8a0c6d7b4",
    "b458a5f7-193a-46f4-97d5-aac510bb5952",
    "b4e8e959-5146-4418-9a89-ffe9def5dcbd",
    "ba41442b-e46e-41a1-870c-ff313b6a0c8c",
    "bb3aa1b6-db01-4610-9689-9d0101e92d17",
    "c986f32b-b752-45d2-ae81-9ba9b48289f7",
    "d9ae0c7d-e2bb-4ee4-94e2-4870ae5eb80a",
    "df3a0e29-0fc7-4cd5-838f-326822e098fc",
    "e6c4fbe6-b338-4ddb-9994-eb3097535278",
    "e7049e4f-fb78-4208-aa24-907ebf9a9ae9",
    "ef47fd93-925a-42f4-be44-7b302cee44d3",
    "f36e2c39-34cb-4b03-a2f9-68cb33bdbebe"
]

missing_directories = set(directories) - set(ids)

print("Directories not in the IDs list:")
for directory in sorted(missing_directories):
    print(directory)

print(f"\nTotal number of missing directories: {len(missing_directories)}")
