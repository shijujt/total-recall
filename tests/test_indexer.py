
import pytest

from ir.indexer import MarkdownSectionParser


@pytest.fixture
def parser(tmp_path):
    return MarkdownSectionParser(service_name="lambda", max_tokens=400)


def write_md(tmp_path, content, filename="test.md"):
    p = tmp_path / filename
    p.write_text(content, encoding="utf-8")
    return str(p)


# ---------------------------------------------------------------------------
# H2 strategy
# ---------------------------------------------------------------------------


class TestH2Strategy:
    def test_returns_one_chunk_per_h2(self, parser, tmp_path):
        content = "# Title\n\n## Section One\n\nsome text\n\n## Section Two\n\nmore text\n"
        path = write_md(tmp_path, content)
        chunks = parser.parse_file(path)
        assert len(chunks) == 2

    def test_metadata_fields_present(self, parser, tmp_path):
        content = "## Section\n\ntext\n"
        path = write_md(tmp_path, content)
        chunks = parser.parse_file(path)
        meta = chunks[0]["metadata"]
        assert "service" in meta
        assert "file" in meta
        assert "section_title" in meta
        assert "heading_level" in meta
        assert "chunk_id" in meta

    def test_metadata_values_correct(self, parser, tmp_path):
        content = "## My Section\n\ntext\n"
        path = write_md(tmp_path, content, filename="lambda-config.md")
        chunks = parser.parse_file(path)
        meta = chunks[0]["metadata"]
        assert meta["service"] == "lambda"
        assert meta["file"] == "lambda-config.md"
        assert meta["section_title"] == "My Section"
        assert meta["heading_level"] == 2

    def test_h2_preferred_over_h3(self, parser, tmp_path):
        content = "## H2 Section\n\n### H3 Sub\n\ntext\n\n## Another H2\n\nmore\n"
        path = write_md(tmp_path, content)
        chunks = parser.parse_file(path)
        assert all(c["metadata"]["heading_level"] == 2 for c in chunks)
        assert len(chunks) == 2

    def test_html_stripped_from_title(self, parser, tmp_path):
        content = '## Foo <a href="#">bar</a>\n\ntext\n'
        path = write_md(tmp_path, content)
        chunks = parser.parse_file(path)
        assert chunks[0]["metadata"]["section_title"] == "Foo bar"

    def test_empty_file_returns_empty_list(self, parser, tmp_path):
        path = write_md(tmp_path, "   \n  ")
        chunks = parser.parse_file(path)
        assert chunks == []

    def test_only_h1_triggers_fallback(self, parser, tmp_path):
        # Only H1 means levels > 1 is empty → fallback called
        # With no bold patterns and short content, token split should run
        content = "# Top Level\n\n" + "word " * 10
        path = write_md(tmp_path, content)
        chunks = parser.parse_file(path)
        # Fallback should produce at least one chunk
        assert len(chunks) >= 1

    def test_no_headings_triggers_fallback(self, parser, tmp_path):
        content = "just plain text without any headings\n"
        path = write_md(tmp_path, content)
        chunks = parser.parse_file(path)
        assert len(chunks) >= 1

    def test_chunk_text_contains_service_and_section(self, parser, tmp_path):
        content = "## Execution Role\n\nconfigure the role\n"
        path = write_md(tmp_path, content)
        chunks = parser.parse_file(path)
        text = chunks[0]["text"]
        assert "Lambda" in text
        assert "Execution Role" in text

    def test_chunk_id_unique_per_chunk(self, parser, tmp_path):
        content = "## A\n\ntext\n\n## B\n\nmore\n\n## C\n\neven more\n"
        path = write_md(tmp_path, content)
        chunks = parser.parse_file(path)
        ids = [c["metadata"]["chunk_id"] for c in chunks]
        assert len(ids) == len(set(ids))


# ---------------------------------------------------------------------------
# Bold section strategy
# ---------------------------------------------------------------------------


class TestBoldStrategy:
    def test_bold_sections_returned(self, parser, tmp_path):
        content = "intro\n\n+ **Timeout**\n\nsome timeout text\n\n+ **Memory**\n\nsome memory text\n"  # noqa: E501
        path = write_md(tmp_path, content)
        chunks = parser.parse_file(path)
        levels = [c["metadata"]["heading_level"] for c in chunks]
        assert all(lv == "bold_section" for lv in levels)

    def test_bold_section_title_extracted(self, parser, tmp_path):
        content = "no headings\n\n+ **Timeout**\n\ntext\n\n+ **Memory**\n\nmore\n"
        path = write_md(tmp_path, content)
        chunks = parser.parse_file(path)
        titles = [c["metadata"]["section_title"] for c in chunks]
        assert "Timeout" in titles
        assert "Memory" in titles

    def test_single_bold_falls_through_to_token(self, parser, tmp_path):
        # Only one bold match → _split_by_bold_sections returns [] → token split
        content = "no headings\n\n+ **OnlyOne**\n\nsome text here\n"
        path = write_md(tmp_path, content)
        chunks = parser.parse_file(path)
        assert all(c["metadata"]["heading_level"] == "token_split" for c in chunks)


# ---------------------------------------------------------------------------
# Token split strategy
# ---------------------------------------------------------------------------


class TestTokenStrategy:
    def test_chunk_count_correct(self, tmp_path):
        parser = MarkdownSectionParser(service_name="s3", max_tokens=400)
        content = "word " * 1200  # exactly 1200 words
        path = write_md(tmp_path, content)
        chunks = parser.parse_file(path)
        assert len(chunks) == 3

    def test_heading_level_is_token_split(self, tmp_path):
        parser = MarkdownSectionParser(service_name="s3", max_tokens=400)
        content = "word " * 500
        path = write_md(tmp_path, content)
        chunks = parser.parse_file(path)
        assert all(c["metadata"]["heading_level"] == "token_split" for c in chunks)

    def test_last_chunk_smaller_than_max(self, tmp_path):
        parser = MarkdownSectionParser(service_name="s3", max_tokens=400)
        content = "word " * 500  # 500 words → chunks of 400 and 100
        path = write_md(tmp_path, content)
        chunks = parser.parse_file(path)
        assert len(chunks) == 2
        last_words = chunks[-1]["text"].split()
        # Last chunk should contain fewer than 400 content words
        assert len(last_words) < 400

    def test_section_title_matches_chunk_n_pattern(self, tmp_path):
        parser = MarkdownSectionParser(service_name="s3", max_tokens=400)
        content = "word " * 800
        path = write_md(tmp_path, content)
        chunks = parser.parse_file(path)
        for i, chunk in enumerate(chunks, start=1):
            assert chunk["metadata"]["section_title"] == f"Chunk {i}"


# ---------------------------------------------------------------------------
# _format_chunk
# ---------------------------------------------------------------------------


class TestFormatChunk:
    def test_contains_service_name_capitalized(self, parser, tmp_path):
        result = parser._format_chunk(
            filepath="/tmp/test.md",
            section_title="My Section",
            section_text="some content",
        )
        assert "AWS Lambda" in result

    def test_contains_section_title(self, parser, tmp_path):
        result = parser._format_chunk(
            filepath="/tmp/test.md",
            section_title="Execution Role",
            section_text="content here",
        )
        assert "Execution Role" in result

    def test_contains_filename(self, parser, tmp_path):
        result = parser._format_chunk(
            filepath="/tmp/some-file.md",
            section_title="Section",
            section_text="body",
        )
        assert "some-file.md" in result
