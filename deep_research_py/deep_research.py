from typing import List, Dict, TypedDict, Optional
import asyncio
import os
import openai
import time
from firecrawl import FirecrawlApp
from deep_research_py.ai.providers import trim_prompt, generate_completions
from deep_research_py.prompt import system_prompt
from deep_research_py.common.logging import log_event, log_error, log_warning
from deep_research_py.common.token_cunsumption import (
    parse_ollama_token_consume,
    parse_openai_token_consume,
)
from deep_research_py.utils import get_service
from deep_research_py.utils.retry import retry_with_exponential_backoff
import json
from pydantic import BaseModel


class SearchResponse(TypedDict):
    data: List[Dict[str, str]]


class ResearchResult(TypedDict):
    learnings: List[str]
    visited_urls: List[str]


class SerpQuery(BaseModel):
    query: str
    research_goal: str


class Firecrawl:
    """Simple wrapper for Firecrawl SDK with rate limiting."""

    def __init__(self, api_key: str = "", api_url: Optional[str] = None):
        self.app = FirecrawlApp(api_key=api_key, api_url=api_url)
        self.last_request_time = 0
        self.min_request_interval = 2.0  # Increased minimum time between requests
        self.max_retries = 5  # Increased max retries
        self.base_delay = 2

    async def _wait_for_rate_limit(self):
        """Implements rate limiting between requests."""
        now = time.time()
        time_since_last_request = now - self.last_request_time
        if time_since_last_request < self.min_request_interval:
            await asyncio.sleep(self.min_request_interval - time_since_last_request)
        self.last_request_time = time.time()

    @retry_with_exponential_backoff(max_retries=5, base_delay=2.0, retry_on=(429,))
    async def search(
        self, query: str, timeout: int = 15000, limit: int = 5
    ) -> SearchResponse:
        """Search using Firecrawl SDK with retry logic."""
        await self._wait_for_rate_limit()
        
        # Run the synchronous SDK call in a thread pool
        response = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.app.search(
                query=query,
            ),
        )

        # Handle the response format from the SDK
        if isinstance(response, dict) and "data" in response:
            return response
        elif isinstance(response, dict) and "success" in response:
            return {"data": response.get("data", [])}
        elif isinstance(response, list):
            formatted_data = []
            for item in response:
                if isinstance(item, dict):
                    formatted_data.append(item)
                else:
                    formatted_data.append(
                        {
                            "url": getattr(item, "url", ""),
                            "markdown": getattr(item, "markdown", "")
                            or getattr(item, "content", ""),
                            "title": getattr(item, "title", "")
                            or getattr(item, "metadata", {}).get("title", ""),
                        }
                    )
            return {"data": formatted_data}
        else:
            print(f"Unexpected response format from Firecrawl: {type(response)}")
            return {"data": []}


# Initialize Firecrawl
firecrawl = Firecrawl(
    api_key=os.environ.get("FIRECRAWL_API_KEY", ""),
    api_url=os.environ.get("FIRECRAWL_BASE_URL"),
)


class SerpQueryResponse(BaseModel):
    queries: List[SerpQuery]


async def generate_serp_queries(
    query: str,
    client: openai.OpenAI,
    model: str,
    num_queries: int = 3,
    learnings: Optional[List[str]] = None,
) -> List[SerpQuery]:
    """Generate SERP queries based on user input and previous learnings."""

    prompt = f"""Given the following prompt from the user, generate a list of SERP queries to research the topic. Return a JSON object with a 'queries' array field containing {num_queries} queries (or less if the original prompt is clear). Each query object should have 'query' and 'research_goal' fields. Make sure each query is unique and not similar to each other: <prompt>{query}</prompt>"""

    if learnings:
        prompt += f"\n\nHere are some learnings from previous research, use them to generate more specific queries: {' '.join(learnings)}"

    response = await generate_completions(
        client=client,
        model=model,
        messages=[
            {"role": "system", "content": system_prompt()},
            {"role": "user", "content": prompt},
        ],
        format=SerpQueryResponse.model_json_schema(),
    )

    try:
        result = SerpQueryResponse.model_validate_json(
            response.choices[0].message.content
        )
        parse_openai_token_consume("generate_serp_queries", response)

        queries = result.queries if result.queries else []
        log_event(f"Generated {len(queries)} SERP queries for research query: {query}")
        log_event(f"Got queries: {queries}")
        return queries[:num_queries]
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON response: {e}")
        print(f"Raw response: {response.choices[0].message.content}")
        log_error(
            f"Failed to parse JSON response for query: {query}, raw response: {response.choices[0].message.content}"
        )
        return []


class SerpResultResponse(BaseModel):
    learnings: List[str]
    followUpQuestions: List[str]


async def process_serp_result(
    query: str,
    search_result: SearchResponse,
    client: openai.OpenAI,
    model: str,
    num_learnings: int = 3,
    num_follow_up_questions: int = 3,
) -> Dict[str, List[str]]:
    """Process search results to extract learnings and follow-up questions."""

    contents = [
        trim_prompt(item.get("markdown", ""), 25_000)
        for item in search_result["data"]
        if item.get("markdown")
    ]

    # Create the contents string separately
    contents_str = "".join(f"<content>\n{content}\n</content>" for content in contents)

    prompt = (
        f"Given the following contents from a SERP search for the query <query>{query}</query>, "
        f"generate a list of learnings from the contents. Return a JSON object with 'learnings' "
        f"and 'followUpQuestions' keys with array of strings as values. Include up to {num_learnings} learnings and "
        f"{num_follow_up_questions} follow-up questions. The learnings should be unique, "
        "concise, and information-dense, including entities, metrics, numbers, and dates.\n\n"
        f"<contents>{contents_str}</contents>"
    )

    response = await generate_completions(
        client=client,
        model=model,
        messages=[
            {"role": "system", "content": system_prompt()},
            {"role": "user", "content": prompt},
        ],
        format=SerpResultResponse.model_json_schema(),
    )

    try:
       
        result = SerpResultResponse.model_validate_json(
            response.choices[0].message.content
        )
        parse_openai_token_consume("process_serp_result", response)

        log_event(
            f"Processed SERP results for query: {query}, found {len(result.learnings)} learnings and {len(result.followUpQuestions)} follow-up questions"
        )
        log_event(
            f"Got learnings: {len(result.learnings)} and follow-up questions: {len(result.followUpQuestions)}"
        )
        return {
            "learnings": result.learnings[:num_learnings],
            "followUpQuestions": result.followUpQuestions[:num_follow_up_questions],
        }
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON response: {e}")
        print(f"Raw response: {response.choices[0].message.content}")
        log_error(
            f"Failed to parse SERP results for query: {query}, raw response: {response.choices[0].message.content}"
        )
        return {"learnings": [], "followUpQuestions": []}


class FinalReportResponse(BaseModel):
    reportMarkdown: str


async def write_final_report(
    prompt: str,
    learnings: List[str],
    visited_urls: List[str],
    client: openai.OpenAI,
    model: str,
) -> str:
    """Generate final report based on all research learnings."""
    
    log_event("Starting to generate final report...")
    log_event(f"Processing {len(learnings)} learnings from {len(visited_urls)} sources")

    learnings_string = trim_prompt(
        "\n".join([f"<learning>\n{learning}\n</learning>" for learning in learnings]),
        150_000,
    )

    user_prompt = (
        f"Given the following prompt from the user, write a final report on the topic using "
        f"the learnings from research. Return a JSON object with a 'reportMarkdown' field "
        f"containing a detailed markdown report (aim for 3+ pages). Include ALL the learnings "
        f"from research:\n\n<prompt>{prompt}</prompt>\n\n"
        f"Here are all the learnings from research:\n\n<learnings>\n{learnings_string}\n</learnings>"
    )

    log_event("Generating final report content...")
    messages = [{"role": "user", "content": user_prompt}]
    
    try:
        response = await generate_completions(
            client=client,
            model=model,
            messages=messages,
            format={"type": "json_object"}
        )

        log_event("Final report generated successfully")
        
        if get_service() == "ollama":
            return response["message"]["content"]
        else:
            return response.choices[0].message.content
    except Exception as e:
        log_error(f"Error generating final report: {str(e)}")
        raise


async def deep_research(
    query: str,
    breadth: int,
    depth: int,
    concurrency: int,
    client: openai.OpenAI,
    model: str,
    learnings: List[str] = None,
    visited_urls: List[str] = None,
) -> ResearchResult:
    """
    Main research function that recursively explores a topic.
    """
    learnings = learnings or []
    visited_urls = visited_urls or []

    log_event(f"Starting research for query: {query}")
    log_event(f"Current depth: {depth}, breadth: {breadth}, concurrency: {concurrency}")

    # Generate search queries
    log_event("Generating search queries...")
    serp_queries = await generate_serp_queries(
        query=query,
        client=client,
        model=model,
        num_queries=breadth,
        learnings=learnings,
    )

    if not serp_queries:
        log_warning("No search queries generated, research may be incomplete")

    # Create a semaphore to limit concurrent requests
    semaphore = asyncio.Semaphore(concurrency)

    async def process_query(serp_query: SerpQuery, query_index: int) -> ResearchResult:
        async with semaphore:
            try:
                log_event(f"Processing query {query_index + 1}/{len(serp_queries)}: {serp_query.query}")
                
                # Add delay based on query index to stagger requests
                await asyncio.sleep(query_index * 2)
                
                # Search for content
                log_event(f"Searching content for: {serp_query.query}")
                result = await firecrawl.search(
                    serp_query.query, timeout=15000, limit=5
                )

                # Collect new URLs
                new_urls = [
                    item.get("url") for item in result["data"] if item.get("url")
                ]
                if new_urls:
                    log_event(f"Found {len(new_urls)} new sources")

                # Calculate new breadth and depth for next iteration
                new_breadth = max(1, breadth // 2)
                new_depth = depth - 1

                # Process the search results
                log_event("Processing search results...")
                new_learnings = await process_serp_result(
                    query=serp_query.query,
                    search_result=result,
                    num_follow_up_questions=new_breadth,
                    client=client,
                    model=model,
                )

                all_learnings = learnings + new_learnings["learnings"]
                all_urls = visited_urls + new_urls

                # If we have more depth to go, continue research with delay
                if new_depth > 0:
                    log_event(
                        f"Research level {new_depth}: Breadth={new_breadth}"
                    )
                    # Add delay before going deeper
                    await asyncio.sleep(3)

                    next_query = f"""
                    Previous research goal: {serp_query.research_goal}
                    Follow-up research directions: {" ".join(new_learnings["followUpQuestions"])}
                    """.strip()

                    return await deep_research(
                        query=next_query,
                        breadth=new_breadth,
                        depth=new_depth,
                        concurrency=max(1, concurrency - 1),
                        learnings=all_learnings,
                        visited_urls=all_urls,
                        client=client,
                        model=model,
                    )

                return {"learnings": all_learnings, "visited_urls": all_urls}

            except Exception as e:
                if "Timeout" in str(e):
                    log_error(f"Timeout error for query '{serp_query.query}': {str(e)}")
                else:
                    log_error(f"Error processing query '{serp_query.query}': {str(e)}")
                return {"learnings": [], "visited_urls": []}

    # Process queries with index for staggered delays
    log_event(f"Processing {len(serp_queries)} queries in parallel...")
    results = await asyncio.gather(*[
        process_query(query, idx) 
        for idx, query in enumerate(serp_queries)
    ])

    # Combine all results
    all_learnings = list(
        set(learning for result in results for learning in result["learnings"])
    )
    log_event(f"Collected {len(all_learnings)} unique findings")

    all_urls = list(set(url for result in results for url in result["visited_urls"]))
    log_event(f"Total unique sources: {len(all_urls)}")

    return {"learnings": all_learnings, "visited_urls": all_urls}
