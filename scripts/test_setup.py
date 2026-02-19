"""
Smoke test: verify all services are running and connected.

    python -m scripts.test_setup
"""

import sys


def main():
    print("=== Mushroom AI — Setup Verification ===\n")
    all_ok = True

    # 1. PostgreSQL
    print("1. PostgreSQL + pgvector...")
    try:
        from db.connection import check_connection
        if check_connection():
            print("   ✓ Connected, pgvector extension enabled")
        else:
            print("   ✗ Connection failed")
            all_ok = False
    except Exception as e:
        print(f"   ✗ {e}")
        all_ok = False

    # 2. Ollama
    print("2. Ollama LLM...")
    try:
        from llm.client import completion
        result = completion("Say 'hello' in one word.")
        print(f"   ✓ Response: {result[:80]}")
    except Exception as e:
        print(f"   ✗ {e}")
        print("   Check: curl http://localhost:11434/api/tags")
        all_ok = False

    # 3. Instructor structured output
    print("3. Instructor structured output...")
    try:
        from pydantic import BaseModel, Field
        from llm.client import structured_completion

        class TestOut(BaseModel):
            name: str = Field(description="Name of a mushroom")
            is_edible: bool = Field(description="Whether it is edible")

        result = structured_completion(
            "Tell me about the common button mushroom.",
            response_model=TestOut,
        )
        print(f"   ✓ Parsed: name={result.name}, edible={result.is_edible}")
    except Exception as e:
        print(f"   ✗ {e}")
        all_ok = False

    # 4. MLflow
    print("4. MLflow tracking server...")
    try:
        import urllib.request
        resp = urllib.request.urlopen(
            "http://localhost:5000/api/2.0/mlflow/experiments/search?max_results=1"
        )
        if resp.status == 200:
            print("   ✓ Reachable at http://localhost:5000")
        else:
            print(f"   ✗ Status {resp.status}")
            all_ok = False
    except Exception as e:
        print(f"   ✗ {e}")
        all_ok = False

    # Summary
    print()
    if all_ok:
        print("All checks passed! Ready to start development.")
    else:
        print("Some checks failed. Fix the issues above before proceeding.")
        sys.exit(1)


if __name__ == "__main__":
    main()
