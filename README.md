# Semantic Search API

This is a POC for a semantic search API implementation using OpenAI models.
The goal is to demonstrate how semantic search enables natural language queries compared to keyword search and traditional methods.

**[Dataset Link](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023)**

**Key Assumptions**

- The system should be easily runnable in any local environment.
  - Users are expected to have Python, npm, and Docker installed and configured, with internet access.
  - Design choices prioritize ease of local deployment over cloud optimization.
  - Cloud deployment (AWS/Azure/GCP) would require platform-specific optimizations:
    - For AWS, the system could be deployed to ECS or EKS, but would need modifications for Lambda deployment.
    - Cloud environments could leverage managed vector database services, requiring code adaptations.
    - You can view a high level proposed architecture on AWS [DEPLOYMENT_ARCHITECTURE in draw.io](deployment_architecture_diagram.drawio), one on ECS the other using Lambda.
- The POC addresses E-Commerce platform requirements
  - Performance is critical for supporting high concurrent user loads.
  - Data ingestion must scale to handle frequent product updates and large catalog changes.

## Repository Structure

```text
.
├── Dockerfile
├── README.md
├── deployment_architecture_diagram.drawio    # AWS deployment architectures
├── code_architecture_diagram.drawio         # System component interactions
├── dockercompose.yml
├── meta_Amazon_Fashion.jsonl
├── pyproject.toml
├── semantic_search_api
│   ├── __init__.py
│   ├── api
│   │   ├── __init__.py
│   │   ├── main.py
│   │   └── routes
│   │       ├── __init__.py
│   │       └── query.py
│   ├── config.py
│   ├── database
│   │   ├── __init__.py
│   │   ├── data_ingestion.py
│   │   ├── database_query.py
│   │   └── tables.py
│   ├── local_database
│   │   ├── __init__.py
│   │   └── database_setup.py
│   ├── main.py
│   └── oai_operations
│       ├── __init__.py
│       ├── rerank_results.py
│       └── search_workflow.py
├── tests
│   ├── __init__.py
│   ├── conftest.py
│   ├── small_test_data.jsonl
│   ├── test_api.py
│   ├── test_data_ingestion.py
│   └── test_queries.py
└── uv.lock

8 directories, 29 files

```

High level code architecture diagram: [CODE ARCHITECTURE in draw.io](code_architecture_diagram.drawio)

The python code lives within the `/semantic_search_api` directory:

- `/semantic_search_api/local_database/` - Database setup and configuration scripts
- `/semantic_search_api/database/` - Core database operations (queries, ingestion)
- `/semantic_search_api/api/` - API endpoints and routing logic
- `/semantic_search_api/oai_operations/` - OpenAI API integration
- `/tests/` - Integration and unit tests with sample data

`/semantic_search_api/config.py`  - Is responsible for loading application wide configurations from a .env file.
This would be changed in a production system to read from a secret store and environment variables. All other parts of the application rely on the configurations created within this file.

`/semantic_search_api/main.py` - This is the file holds the API configuration using [FastAPI](https://fastapi.tiangolo.com). The `/semantic_search_api/api/` directory holds the rest of the API configuration. This currently has very small amount and all of it could just live within the `main.py` from the application root, but I intended to show the structure under which API functionality could be scaled out with a clear pattern.

`/semantic_search_api/database` - This directory holds everything that is related to database interactions. Embedding table definition, data ingestion and data query.

`/semantic_search_api/oai_operations` - This directory holds modules that centre around OpenAI api calls.

`pyproject.toml` is where the dependencies are stored. The dependency management is done by [UV](https://docs.astral.sh/uv/), and through this linting and formatting is handled by [Ruff](https://docs.astral.sh/ruff/).

## Running the project

### Running the Project Locally in Docker

This assumes that you have docker desktop installed and are able to build and run linux based docker images. If this doesn't sound right please follow these guides to setup.
[Docker Desktop Install](https://docs.docker.com/desktop/setup/install/mac-install/) (pick the correct platform). And then return to this guide.

#### 1. CD into the project folder

```bash
cd <project_repo>
```

#### 2. Create your `.env` file at the folder root

You should **keep** the provided default values for a quick startup and provide valid OpenAI api credentials.

```text
POSTGRES_USER = postgres
POSTGRES_PASSWORD = password99
POSTGRES_DB = vectordb

DATABASE_HOST = "localhost"

OPENAI_API_ORGANIZATION=
OPENAI_API_PROJECT_ID=
OPENAI_API_KEY=

VECTOR_SIZE=3072

EMBEDDING_MODEL="text-embedding-3-large"
RE_RANK_MODEL="gpt-4o-2024-11-20"
QUERY_REFINEMENT_MODEL="gpt-4o-2024-11-20"
```

#### 3. Start the Project in Docker

```bash
docker compose -f 'dockercompose.yml' up -d --build
```

For viewing container logs in real-time, omit the -d flag:

```bash
docker compose -f 'dockercompose.yml' up --build
```

This builds two Docker images:

1. The project image
2. A Postgres database image with vector search capabilities

The setup script configures the Postgres database for vector operations.

You can access the API documentation at [http://localhost:8000/docs](http://localhost:8000/docs)

The Swagger UI is interactive and allows testing all API endpoints.

At this point you will have a small amount of records in the database that you can try working with.

#### 4. Optional: Load More Data to work with

If you would like to use the API for further data loading, you'll need to place your data into the **my-fastapi-containers** volume.

```bash
docker cp <path to jsonl file> my-fastapi-container:/data
```

At this point you will be able to use the data_load api endpoint, you'll only need to pass your file name without path prefixes.

#### 7. Clean Up Docker setup

```bash
docker-compose -f dockercompose.yml down -v
```

### Running the Project Locally

This will assume you have python 3.12 on your computer. The guide also assumes you're on a Mac or Linux system or WSL (Windows Subsystem for Linux).

#### 1. CD into the project folder

```bash
cd <project_repo>
```

#### 2. Create your `.env` file at the folder root

Please **keep** the provided default values for a quick startup and provide valid OpenAI api credentials.

```text
DATABASE_URL = "postgresql+asyncpg://postgres:password99@localhost:5432/vectordb"

OPENAI_API_ORGANIZATION=""
OPENAI_API_PROJECT_ID=""
OPENAI_API_KEY=""

VECTOR_SIZE=3072

EMBEDDING_MODEL="text-embedding-3-large"
RE_RANK_MODEL="gpt-4o-2024-11-20"
QUERY_REFINEMENT_MODEL="gpt-4o-2024-11-20"
```

#### 3. Create virtual environment and install dependencies

```bash
python -m venv .venv
```

```bash
source .venv/bin/activate
```

```bash
pip install uv
```

```bash
uv sync
```

#### 4. Start a Postgres server in container

```bash
docker-compose up db
```

#### 5. Start the fastAPI local server

In a new terminal window run:

```bash
fastapi dev semantic_search_api/main.py
```

The command will output the localhost address at which you can reach the functioning API Swagger. Most likely [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

#### 6. Data Loading

Data can be loaded into the database either through the api endpoint. In which case you should be passing the entire file path.
Or you can use the data ingestion code as a cli tool.

Check the instructions for the CLI:

```bash
python semantic_search_api/database/data_ingestion.py --help
```

## Considerations

### Vector Database

While there are many vector database options commonly used as LLM knowledge bases (such as Pinecone, ElasticSearch, OpenSearch, and Redis),
this POC uses Postgres because it's:

- Open-source and free to use
- Easily replicable
- Familiar to enterprise environments

Large enterprises working with forward-deployed engineers often prefer avoiding additional managed services due to vendor lock-in concerns. Prototyping with open-source technologies they can self-manage helps address these concerns.

### Data Ingestion

The data ingestion was designed to not load the entire dataset into memory. It loops through each line and sends off batched data to the embedding endpoint and the database.
This was considered useful for the POC because it allows the code to process large datasets without requiring to provision compute that hold those datasets in memory.
This reduces infrastructure costs for the client, but also allows to accelerate the next step of the prototyping of testing the performance of prompts and models on large dataset to be able to infer statistically significant differences in end to end performance of the system.

### OpenAI api calls

The [OpenAI documentation](https://platform.openai.com/docs/api-reference/chat/create) indicates that there should be some level of prompt injection defence built into the API for chat completions, by utilising the role variables. The models should always follow the developer prompt regardless of the user prompt. In the code all chat completion calls use the user role to pass dynamic input, which might be prone to injection. There could be further guardrails and moderations used later.

### Testing

The test suite implements end-to-end testing functionality. Running the tests creates a temporary copy of the production database schema, which is automatically cleaned up after testing.

Current limitations:

- OpenAI API calls consume credits during tests. Frequently tested functions should use mocked API calls to reduce costs.
- Test coverage is not complete but demonstrates various testing strategies that could be applied.
- Even with partial coverage, the tests provide confidence for code refactoring.

## Next Steps and Improvements to Consider

### Prompt and model Selection Refinement

There some coding work to be done to parametrise the prompts used through the system.

- Start the API with a given set of models in the config and prompts.
- Create a set of queries expected from the system.
- Run these queries against the API and save the outputs mapped to the input.
- Repeat the process with different models configured and prompts.
- Evaluate the resulting datasets against each other both by humans and by using an LLM as a judge.

### Improved Re-rank capability

Enable reasoning models for the re-rank to produce higher quality product rankings from the retrieved results. This could be feature only enabled for high value customers within the platform due to its increased cost.

If re-rank is reliable enough allow the mechanism to filter out irrelevant results.

### Improved Results filtering

Currently the database query returns the most similar results it can find. With further testing the this query could be tuned to not return values under a given similarity score.

### Hybrid Retrieval

The database query could be enhanced by combining keyword search into the query. Keywords could either be passed in through explicit filters from the platform, or the text query could be passed through an LLM to pick from a list of possible keywords to use for filters passed to the database query.

### Multimodal Embedding

The dataset contains videos and images. These could be fed through a multimodal LLM to produce a description of them and add that those description into the text that is being embedded at the point of ingestion.

The other option would be to utilise a multimodal embedding model, but based on current OpenAI documentation they don't currently offer one.
