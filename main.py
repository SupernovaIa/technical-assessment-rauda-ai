import os
import json
import logging
import dotenv
import pandas as pd
from tqdm import tqdm
from langchain.schema import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

# Configure logging to record errors in 'error.log'
logging.basicConfig(
    level=logging.ERROR,
    filename="error.log",
    filemode="a",
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Load environment variables
dotenv.load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Initialize the language model
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.3, openai_api_key=api_key)

def evaluate_reply(ticket, reply):
    """
    Sends a customer support ticket and reply to a LLM for evaluation using LangChain.
    
    Evaluates the response based on content and format.
    
    Returns a JSON with:
    - content_score: integer (1-5)
    - content_explanation: string
    - format_score: integer (1-5)
    - format_explanation: string
    """
    prompt = f"""
    Given the following customer support ticket and reply, evaluate the quality of the response based on content and format.
    
    Ticket: "{ticket}"
    
    Reply: "{reply}"
    
    Provide two scores:
    - Content Score (1-5): How well does the reply address the customer's concern? Evaluate relevance, correctness and completeness.
    - Format Score (1-5): How clear, professional, and well-structured is the response? Evaluate clarity, structure, grammar/spelling.
    
    Additionally, provide short textual explanations for both scores.
    
    Output format (JSON):
    {{
        "content_score": <integer>,
        "content_explanation": "<string>",
        "format_score": <integer>,
        "format_explanation": "<string>"
    }}
    """
    
    try:
        response = llm.invoke([
            SystemMessage(content="You are an AI trained to evaluate customer service replies."),
            HumanMessage(content=prompt)
        ])
        result = response.content
        
        try:
            evaluation = json.loads(result)
        except json.JSONDecodeError:
            raise ValueError("The response is not a valid JSON")
        
        return evaluation

    except Exception as e:
        logging.error(f"Error processing ticket: {e}")
        print(f"Error processing ticket: {e}")
        return {
            "content_score": None,
            "content_explanation": "Error",
            "format_score": None,
            "format_explanation": "Error"
        }

def process_tickets(input_csv="tickets.csv", output_csv="tickets_evaluated.csv"):
    """
    Reads the input CSV file, evaluates each ticket-reply pair using evaluate_reply,
    and writes the results to the output CSV file.
    
    Returns the output CSV file path.
    """
    # Check if the input CSV file exists
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"The CSV file '{input_csv}' does not exist.")
    
    # Load the data
    df = pd.read_csv(input_csv)
    
    # Verify that the required columns exist
    if "ticket" not in df.columns or "reply" not in df.columns:
        raise ValueError("The CSV file must contain the columns 'ticket' and 'reply'.")
    
    # Evaluate each row in the DataFrame
    evaluations = []
    for index, row in tqdm(df.iterrows(), total=len(df)):
        if pd.isna(row["ticket"]) or pd.isna(row["reply"]):
            evaluations.append({
                "content_score": None,
                "content_explanation": "Missing data",
                "format_score": None,
                "format_explanation": "Missing data"
            })
        else:
            ticket_text = str(row["ticket"]).strip()
            reply_text = str(row["reply"]).strip()
            eval_result = evaluate_reply(ticket_text, reply_text)
            evaluations.append(eval_result)
    
    # Convert evaluation results to a DataFrame
    eval_df = pd.DataFrame(evaluations)
    
    # Combine the evaluations with the original DataFrame
    df_result = pd.concat([df, eval_df], axis=1)
    
    # Save the combined results to a new CSV file
    df_result.to_csv(output_csv, index=False)
    return output_csv

if __name__ == "__main__":
    output = process_tickets()
    print(f"Evaluation completed and saved in '{output}'.")
