import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from data_prep import EmailToWordCountTransformer


def test_lowercase():
    # Arrange
    transformer = EmailToWordCountTransformer(input_type="text")
    email_text = "Hello World THIS IS A TEST"

    # Act
    result = transformer.transform([email_text])

    # Assert
    words = result[0].keys()
    assert all(word == word.lower() for word in words), \
        "All words should be lowercase"


def test_replace_urls():
    # Arrange
    transformer = EmailToWordCountTransformer(input_type="text")
    email_text = "visit http://example.com for more info"

    # Act
    result = transformer.transform([email_text])

    # Assert
    assert "url" in result[0], "URL should be replaced with 'url'"


def test_replace_numbers():
    # Arrange
    transformer = EmailToWordCountTransformer(input_type="text")
    email_text = "buy 3 items for only 99 dollars"

    # Act
    result = transformer.transform([email_text])

    # Assert
    assert "number" in result[0], "Numbers should be replaced with 'number'"
    assert "3" not in result[0], "Original number should not exist"
    assert "99" not in result[0], "Original number should not exist"