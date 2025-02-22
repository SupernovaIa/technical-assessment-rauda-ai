# LLM-based ticket reply evaluation

## 1. Introduction

This repository contains a technical assessment for a selection process at Rauda AI. The assessment involves evaluating customer support ticket replies using GPT-4o-mini, a highly efficient and cost-effective model.

Each reply is assessed on two dimensions: `Content` (relevance, correctness, completeness) and `Format` (clarity, structure, grammar/spelling). The evaluation process generates a new CSV file, tickets_evaluated.csv, with four additional columns:

- content_score (1-5)
- content_explanation
- format_score (1-5)
- format_explanation

The implementation is built with LangChain, which simplifies LLM integration and enhances scalability.

## 2. Files

- **README.md** – This file.
- **requirements.txt** – List of project dependencies.
- **tickets.csv** – Input CSV file containing the columns:
  - `ticket` (the customer’s message)
  - `reply` (the AI-generated response)
- **main.py** – Python script that reads `tickets.csv`, evaluates each ticket-reply pair using an LLM, and writes the results to `tickets_evaluated.csv`.
- **test_main.py** – A small test suite using pytest covering core functions.
- **tickets_evaluated.csv** – The output CSV file with additional columns:
  - `content_score`
  - `content_explanation`
  - `format_score`
  - `format_explanation`

## 3. Dependencies

This project uses Python 3.12 and depends on the following Python packages:

- **[langchain (0.3.19)](https://python.langchain.com/docs/)** – Framework for building applications with LLMs.  
- **[langchain_openai (0.3.6)](https://python.langchain.com/docs/integrations/providers/openai/)** – OpenAI integration with LangChain.  
- **[pandas (2.2.3)](https://pandas.pydata.org/docs/)** – Data manipulation and CSV processing.  
- **[tqdm (4.67.1)](https://tqdm.github.io/)** – Progress bar visualization for processing tasks.  
- **[python-dotenv (1.0.1)](https://github.com/theskumar/python-dotenv)** – Environment variable management.  
- **[pytest (8.3.4)](https://docs.pytest.org/en/stable/)** – Testing framework for Python.  
 


All dependencies are listed in the [requirements.txt](requirements.txt) file.

## 4. Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/SupernovaIa/technical-assessment-rauda-ai.git
   cd technical-assessment-rauda-ai
   ```

2. **Create and activate a virtual environment (optional)**

   - **Using Python's built-in venv:**

     ```bash
     python -m venv venv
     source venv/bin/activate  # On Windows: venv\Scripts\activate
     ```

   - **Using Conda:**

     ```bash
     conda create --name env_name python=3.12
     conda activate env_name
     ```

3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Environment Variables**

   Create a `.env` file in the project root directory (this file should not be committed) and add your OpenAI API key:

   ```ini
   OPENAI_API_KEY=your_api_key_here
   ```

## 5. Running the code

To run the main evaluation script, execute:

```bash
python main.py
```

The script will:

1. Read `tickets.csv`.
2. For each row, send the ticket and reply to the LLM with a prompt to evaluate the reply’s content and format.
3. Parse the LLM’s output and extract scores and explanations.
4. Write the results to `tickets_evaluated.csv`.

## 6. Running the tests

A small test suite is included using pytest. To run the tests, execute:

```bash
pytest test_main.py
```

The tests cover:
- Evaluating a single reply using a dummy LLM.
- Processing an input CSV and generating an evaluated CSV.
