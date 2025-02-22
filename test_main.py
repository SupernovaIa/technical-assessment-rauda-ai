import os
import pandas as pd
import pytest
from main import evaluate_reply, process_tickets

# Dummy classes to simulate the LLM behavior
class DummyResponse:
    def __init__(self, content):
        self.content = content

class DummyLLM:
    def invoke(self, messages):
        dummy_json = (
            '{"content_score": 5, "content_explanation": "Great response", '
            '"format_score": 5, "format_explanation": "Excellent format"}'
        )
        return DummyResponse(dummy_json)

# Patch the global llm instance in main.py to use DummyLLM for testing
@pytest.fixture(autouse=True)
def patch_llm(monkeypatch):
    monkeypatch.setattr("main.llm", DummyLLM())

def test_evaluate_reply():
    ticket = "Test ticket"
    reply = "Test reply"
    result = evaluate_reply(ticket, reply)
    expected = {
        "content_score": 5,
        "content_explanation": "Great response",
        "format_score": 5,
        "format_explanation": "Excellent format"
    }
    assert result == expected

def test_process_tickets(tmp_path):
    # Create a temporary CSV file with sample data
    csv_content = "ticket,reply\n\"Ticket 1\",\"Reply 1\"\n\"Ticket 2\",\"Reply 2\""
    input_csv = tmp_path / "tickets.csv"
    input_csv.write_text(csv_content)
    
    output_csv = tmp_path / "tickets_evaluated.csv"
    
    # Run the process_tickets function using the temporary CSV file
    result_csv = process_tickets(str(input_csv), str(output_csv))
    
    # Check that the output CSV file exists
    assert os.path.exists(result_csv)
    
    # Read the output CSV and verify that it has the expected columns and values
    df = pd.read_csv(result_csv)
    for col in ["content_score", "content_explanation", "format_score", "format_explanation"]:
        assert col in df.columns
    
    # Verify that evaluations were performed using DummyLLM (i.e., fixed dummy values)
    assert (df["content_score"] == 5).all()
    assert (df["format_score"] == 5).all()
