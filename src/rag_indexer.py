import re
import os

class MarkdownSectionParser:
    def __init__(self, service_name="lambda", max_tokens=400):
        self.service_name = service_name
        self.max_tokens = max_tokens

    def parse_file(self, filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        if not content.strip():
            return []

        headings = list(re.finditer(r'^(#{1,6})\s+(.*)', content, re.MULTILINE))

        # If no headings at all → fallback splitting
        if not headings:
            return self._fallback_split(filepath, content)

        # Determine available heading levels > H1
        levels = sorted(set(len(h.group(1)) for h in headings if len(h.group(1)) > 1))

        print(f"Parsing: {filepath}")

        # ONLY H1 present → fallback splitting
        if not levels:
            return self._fallback_split(filepath, content)

        # Prefer H2
        if 2 in levels:
            chunk_level = 2
        else:
            chunk_level = levels[0]

        chunks = []

        for i, heading in enumerate(headings):
            level = len(heading.group(1))

            if level != chunk_level:
                continue

            start = heading.start()
            end = headings[i + 1].start() if i + 1 < len(headings) else len(content)

            section_text = content[start:end].strip()

            raw_title = heading.group(2).strip()
            section_title = re.sub(r'<.*?>', '', raw_title).strip()

            chunk_text = self._format_chunk(
                filepath=filepath,
                section_title=section_title,
                section_text=section_text
            )

            chunks.append({
                "text": chunk_text,
                "metadata": {
                    "service": self.service_name,
                    "file": os.path.basename(filepath),
                    "section_title": section_title,
                    "heading_level": chunk_level
                }
            })

        return chunks


    # -------------------------------
    # FALLBACK SPLITTING
    # -------------------------------

    def _fallback_split(self, filepath, content):
        """
        If no H2 sections exist:
        1) Try split by bold subsection markers
        2) Otherwise split by token window
        """

        bold_chunks = self._split_by_bold_sections(filepath, content)

        if bold_chunks:
            return bold_chunks

        return self._split_by_token_length(filepath, content)


    def _split_by_bold_sections(self, filepath, content):
        """
        Split on patterns like:
        + **Timeout**
        + **Memory**
        """

        pattern = r'\n\+\s+\*\*(.*?)\*\*'

        matches = list(re.finditer(pattern, content))

        if len(matches) < 2:
            return []

        chunks = []

        for i, match in enumerate(matches):

            start = match.start()

            end = matches[i + 1].start() if i + 1 < len(matches) else len(content)

            section_text = content[start:end].strip()

            section_title = match.group(1).strip()

            chunk_text = self._format_chunk(
                filepath=filepath,
                section_title=section_title,
                section_text=section_text
            )

            chunks.append({
                "text": chunk_text,
                "metadata": {
                    "service": self.service_name,
                    "file": os.path.basename(filepath),
                    "section_title": section_title,
                    "heading_level": "bold_section"
                }
            })

        return chunks


    def _split_by_token_length(self, filepath, content):
        """
        Last fallback if document has no structure.
        Splits roughly every N tokens.
        """

        words = content.split()
        chunks = []

        for i in range(0, len(words), self.max_tokens):

            chunk_words = words[i:i+self.max_tokens]
            section_text = " ".join(chunk_words)

            chunk_text = self._format_chunk(
                filepath=filepath,
                section_title=f"Chunk {i//self.max_tokens + 1}",
                section_text=section_text
            )

            chunks.append({
                "text": chunk_text,
                "metadata": {
                    "service": self.service_name,
                    "file": os.path.basename(filepath),
                    "section_title": f"Chunk {i//self.max_tokens + 1}",
                    "heading_level": "token_split"
                }
            })

        return chunks


    def _format_chunk(self, filepath, section_title, section_text):
        return f"""Service: AWS {self.service_name.capitalize()}
File: {os.path.basename(filepath)}
Section: {section_title}

{section_text}
"""


import requests
import json


class LlamaQueryGenerator:

    def __init__(self, model="llama3"):
        self.model = model
        self.url = "http://localhost:11434/api/generate"

    def generate_queries(self, chunk_text, service):

        prompt = f"""
You are generating evaluation queries for an AWS documentation search system.

Given the documentation section below, generate 3 realistic developer questions.

Requirements:
- Questions should sound like real developer questions
- Do not copy text directly
- Keep them concise
- Return ONLY valid JSON

Format:
[
  {{"query": "...", "service": "{service}"}},
  {{"query": "...", "service": "{service}"}},
  {{"query": "...", "service": "{service}"}}
]

Documentation:
{chunk_text[:1200]}
"""

        r = requests.post(
            self.url,
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False
            }
        )

        response_text = r.json()["response"]
        match = re.search(r"\[.*\]", response_text, re.DOTALL)

        if not match:
            print("Failed to find JSON array in LLM output")
            print(response_text)
            return []

        json_text = match.group(0)

        try:
            queries = json.loads(json_text)
            return queries
        except:
            print("Failed to parse LLM output")
            print(response_text)
            return []


import os
import chromadb
from chromadb.utils import embedding_functions
import random

class AwsSvcIndexer:

    def __init__(self, base_docs_path, collection_name=None):
        self.base_docs_path = base_docs_path
        self.indexing_mode = collection_name is not None

        if self.indexing_mode:
            self.client = chromadb.PersistentClient(path="../chroma_db")
            self.collection = self.client.get_or_create_collection(name=collection_name)
        else:
            self.query_generator = LlamaQueryGenerator()
        
        self.parser = MarkdownSectionParser()

    def normalize_service(self, folder_name):

        name = folder_name.lower()
        name = name.replace("amazon-", "")
        name = name.replace("aws-", "")
        name = name.replace("-developer-guide", "")
        name = name.replace("-user-guide", "")
        name = name.replace("-docs", "")
        return name

    def index_services(self, svc_names):
        eval_file = None

        if not self.indexing_mode:
            eval_file = open("../eval_queries_ag.jsonl", "w")

        for svc_folder in svc_names:
            svc_path = os.path.join(self.base_docs_path, svc_folder, "doc_source")
            if not os.path.exists(svc_path):
                print(f"Skipping missing folder: {svc_folder}")
                print(f"Computed doc path: {svc_path}")
                continue

            service_name = self.normalize_service(svc_folder)
            if self.indexing_mode:
                print(f"\nIndexing service: {service_name}")
            else:
                print(f"\nGenerating Queries for service: {service_name}")

            self.parser.service_name = service_name
            for root, _, files in os.walk(svc_path):
                sampled_files = random.sample(files, min(len(files), 20))
                for file in files:
                    if not file.endswith(".md"):
                        continue
                    filepath = os.path.join(root, file)
                    chunks = self.parser.parse_file(filepath)
                    if not chunks:
                        continue
                    
                    if not self.indexing_mode:
                        chunks = random.sample(chunks, min(len(chunks), 1))

                    for chunk in chunks:
                        text = chunk["text"]
                        # -----------------------------
                        # MODE 1 — INDEX TO CHROMA
                        # -----------------------------
                        if self.indexing_mode:
                            self.collection.add(documents=[text], metadatas=[chunk["metadata"]], ids=[chunk["metadata"]["chunk_id"]])

                        # -----------------------------
                        # MODE 2 — GENERATE QUERIES
                        # -----------------------------
                        else:
                            if file in sampled_files:
                                queries = self.query_generator.generate_queries(text, service_name)
                                for q in queries:
                                    record = {
                                        "query": q["query"],
                                        "service": service_name,
                                        "src_doc": file,
                                        "keywords": [chunk["metadata"]["section_title"].lower()]
                                    }
                                    eval_file.write(json.dumps(record) + "\n")

            print(f"Finished indexing: {service_name}")
        print("\nIndexing complete.")
        if eval_file:
            eval_file.close()


svc_names = [
             "aws-lambda-developer-guide",
             "amazon-s3-developer-guide", 
             "amazon-s3-getting-started-guide", 
             "amazon-s3-user-guide",
             "amazon-dynamodb-developer-guide",
             "aws-dynamodb-encryption-docs",
             "amazon-api-gateway-developer-guide",
             "amazon-ecs-developer-guide",
             "amazon-elasticsearch-service-developer-guide",
             "amazon-eventbridge-user-guide",
             "amazon-kendra-developer-guide",
             "amazon-sagemaker-developer-guide",
             "amazon-sns-developer-guide",
             "amazon-sqs-developer-guide",
             "aws-cli-user-guide",
             "aws-glue-developer-guide",
             "aws-secrets-manager-docs",
             "iam-user-guide",
             "amazon-opensearch-service-developer-guide",
             "amazon-elasticache-docs",
             "aws-cloudformation-user-guide",
             "aws-data-pipeline-developer-guide",
             "aws-step-functions-developer-guide"
            ]
svc_path =  "./documents"
c_nm = "aws_docs"
c_nm = None
indexer = AwsSvcIndexer(base_docs_path=svc_path, collection_name=c_nm)
indexer.index_services(svc_names)

