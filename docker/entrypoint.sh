#de TODO: Pull in JSON of definitions from hook-models.json
#de       Or check the integrity of the JSON file and pass filename as parameter

#de TODO: Add JSON loader to dynamically do model-hooks

#de Start the Oblix server
uvicorn oblix.main:app --host 0.0.0.0 --port 8140
