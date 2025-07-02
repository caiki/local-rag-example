import pytest
from unittest.mock import patch, MagicMock
import json
from your_module import extract_entities  # Cambia 'your_module' por el nombre real del módulo

@pytest.fixture
def mock_response():
    return {
        "InvoiceNumber": "INV-12345",
        "Date": "2025-07-01",
        "Amount": "1500.00"
    }

@patch("your_module.client.chat.completions.create")
@patch("your_module.preprocess_text_for_extraction", return_value="IMPORTANT: This is the invoice text.")
@patch("your_module.extract_entities_fallback", return_value={})
def test_extract_entities_success(mock_fallback, mock_preprocess, mock_chat, mock_response):
    mock_content = json.dumps(mock_response)
    mock_chat.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content=mock_content))]
    )

    text = "Sample invoice text with data."
    doc_type = "invoice"
    field_list = ["InvoiceNumber", "Date", "Amount"]
    use_generic_prompt = True

    result = extract_entities(text, doc_type, field_list, use_generic_prompt)

    assert isinstance(result, dict)
    assert result["InvoiceNumber"] == "INV-12345"
    assert result["Date"] == "2025-07-01"
    assert result["Amount"] == "1500.00"
    mock_preprocess.assert_called_once()
    mock_chat.assert_called_once()
    mock_fallback.assert_not_called()


@patch("your_module.client.chat.completions.create", side_effect=Exception("Access denied due to Virtual Network/Firewall rules"))
def test_extract_entities_firewall_error(mock_chat):
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc_info:
        extract_entities(
            text="Some text",
            doc_type="invoice",
            field_list=["InvoiceNumber"],
            use_generic_prompt=True
        )

    assert exc_info.value.status_code == 503
    assert "Azure OpenAI devolvió 403" in exc_info.value.detail
