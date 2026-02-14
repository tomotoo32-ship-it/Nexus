"""
AI Notion Hub — Unified AI Router with Notion Read/Write
Supports: Claude, ChatGPT, Grok, Reddit Search
Triggers: Notion DB, Slack, Discord
"""

import os
import re
import json
import httpx
import asyncio
from typing import Optional
from dotenv import load_dotenv
from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
import anthropic
from openai import AsyncOpenAI
from notion_client import AsyncClient as NotionClient

load_dotenv()

app = FastAPI(title="AI Notion Hub", version="1.0.0")

# ─── Clients ────────────────────────────────────────────────────────────────
notion     = NotionClient(auth=os.getenv("NOTION_TOKEN"))
claude_client = anthropic.AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

GROK_API_KEY       = os.getenv("GROK_API_KEY")
NOTION_DB_ID       = os.getenv("NOTION_COMMAND_DB_ID")     # your Command Center DB
NOTION_KB_PAGE_ID  = os.getenv("NOTION_KB_PAGE_ID", "")    # optional Knowledge Base page
SLACK_BOT_TOKEN    = os.getenv("SLACK_BOT_TOKEN", "")
DISCORD_BOT_TOKEN  = os.getenv("DISCORD_BOT_TOKEN", "")


# ─── Routing Logic ───────────────────────────────────────────────────────────

ROUTING_RULES = {
    "reddit":  ["reddit", "find posts", "what do people think", "community", "opinions on"],
    "grok":    ["latest", "news", "what's happening", "today", "current", "right now", "trending"],
    "chatgpt": ["code", "build", "debug", "script", "function", "class", "bug", "program", "python", "javascript"],
    "claude":  ["write", "draft", "edit", "summarize", "rewrite", "improve", "explain", "analyze", "review"],
}

def route_prompt(prompt: str) -> str:
    """Decide which AI should handle a prompt based on keywords."""
    prompt_lower = prompt.lower()
    for ai, keywords in ROUTING_RULES.items():
        if any(kw in prompt_lower for kw in keywords):
            return ai
    return "claude"  # default


# ─── Notion Helpers ──────────────────────────────────────────────────────────

async def get_notion_page_content(page_id: str) -> str:
    """Fetch all text blocks from a Notion page."""
    try:
        blocks = await notion.blocks.children.list(block_id=page_id)
        text_parts = []
        for block in blocks.get("results", []):
            block_type = block.get("type")
            rich_text = block.get(block_type, {}).get("rich_text", [])
            for rt in rich_text:
                text_parts.append(rt.get("plain_text", ""))
        return "\n".join(text_parts)
    except Exception as e:
        return f"[Could not fetch page: {e}]"

async def get_knowledge_base_context() -> str:
    """Pull your Notion Knowledge Base page as context for AIs."""
    if not NOTION_KB_PAGE_ID:
        return ""
    content = await get_notion_page_content(NOTION_KB_PAGE_ID)
    return f"\n\n--- User's Knowledge Base (Notion) ---\n{content}\n---\n"

async def update_notion_row(page_id: str, output: str, ai_used: str):
    """Write AI response back into the Notion command row."""
    await notion.pages.update(
        page_id=page_id,
        properties={
            "Output": {"rich_text": [{"text": {"content": output[:2000]}}]},  # Notion 2000 char limit per block
            "AI Used": {"select": {"name": ai_used.capitalize()}},
            "Status":  {"select": {"name": "Done"}},
        }
    )

async def update_notion_page_content(page_id: str, new_content: str):
    """Replace all text content on a Notion page (live editing)."""
    # First, get existing blocks and delete them
    blocks = await notion.blocks.children.list(block_id=page_id)
    for block in blocks.get("results", []):
        await notion.blocks.delete(block_id=block["id"])

    # Split content into chunks of 2000 chars (Notion limit per block)
    chunks = [new_content[i:i+2000] for i in range(0, len(new_content), 2000)]

    children = [
        {
            "object": "block",
            "type": "paragraph",
            "paragraph": {"rich_text": [{"type": "text", "text": {"content": chunk}}]}
        }
        for chunk in chunks
    ]
    await notion.blocks.children.append(block_id=page_id, children=children)

async def create_notion_page(title: str, content: str, ai_used: str) -> str:
    """Create a new Notion page in your Command Center DB."""
    response = await notion.pages.create(
        parent={"database_id": NOTION_DB_ID},
        properties={
            "Name":   {"title": [{"text": {"content": title}}]},
            "AI Used": {"select": {"name": ai_used.capitalize()}},
            "Status": {"select": {"name": "Done"}},
        },
        children=[
            {
                "object": "block",
                "type": "paragraph",
                "paragraph": {"rich_text": [{"type": "text", "text": {"content": content[:2000]}}]}
            }
        ]
    )
    return response["id"]

async def find_notion_page_by_name(name: str) -> Optional[str]:
    """Search for a Notion page by title in the Command Center DB."""
    response = await notion.databases.query(
        database_id=NOTION_DB_ID,
        filter={
            "property": "Name",
            "title": {"contains": name}
        }
    )
    results = response.get("results", [])
    if results:
        return results[0]["id"]
    return None


# ─── AI Callers ──────────────────────────────────────────────────────────────

async def call_claude(prompt: str, context: str = "") -> str:
    system = f"You are a helpful AI assistant with access to the user's Notion knowledge base.{context}"
    message = await claude_client.messages.create(
        model="claude-opus-4-5-20251101",
        max_tokens=2048,
        system=system,
        messages=[{"role": "user", "content": prompt}]
    )
    return message.content[0].text

async def call_chatgpt(prompt: str, context: str = "") -> str:
    system = f"You are a helpful AI assistant with access to the user's Notion knowledge base.{context}"
    response = await openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": prompt}
        ],
        max_tokens=2048
    )
    return response.choices[0].message.content

async def call_grok(prompt: str, context: str = "") -> str:
    """Call xAI Grok via their OpenAI-compatible API."""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://api.x.ai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {GROK_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "grok-beta",
                "messages": [
                    {"role": "system", "content": f"You are Grok, good at real-time info and news.{context}"},
                    {"role": "user",   "content": prompt}
                ],
                "max_tokens": 2048
            },
            timeout=30.0
        )
        data = response.json()
        return data["choices"][0]["message"]["content"]

async def call_reddit_search(query: str) -> str:
    """Search Reddit and return top post titles + summaries."""
    clean_query = re.sub(r'reddit|search|find posts|what do people think', '', query, flags=re.IGNORECASE).strip()
    async with httpx.AsyncClient() as client:
        response = await client.get(
            "https://www.reddit.com/search.json",
            params={"q": clean_query, "sort": "relevance", "limit": 5},
            headers={"User-Agent": "AI-Notion-Hub/1.0"},
            timeout=15.0
        )
        data = response.json()
    posts = data.get("data", {}).get("children", [])
    if not posts:
        return "No Reddit results found."

    results = [f"Reddit results for: '{clean_query}'\n"]
    for post in posts:
        p = post["data"]
        results.append(f"• [{p['subreddit']}] {p['title']}\n  Score: {p['score']} | Comments: {p['num_comments']}\n  Link: https://reddit.com{p['permalink']}")
    return "\n".join(results)


# ─── Core Dispatcher ─────────────────────────────────────────────────────────

async def dispatch(prompt: str, source_page_id: Optional[str] = None) -> dict:
    """
    Main entry point:
    1. Check for live-edit intent
    2. Route to correct AI
    3. Fetch Notion context
    4. Call AI
    5. Write output back to Notion
    """
    # --- Live edit detection: "edit [Page Name]: instruction" ---
    edit_match = re.match(r'edit\s+(.+?):\s*(.+)', prompt, re.IGNORECASE)
    if edit_match:
        page_name   = edit_match.group(1).strip()
        instruction = edit_match.group(2).strip()
        page_id = await find_notion_page_by_name(page_name)
        if not page_id:
            return {"error": f"Could not find Notion page: '{page_name}'"}

        existing_content = await get_notion_page_content(page_id)
        edit_prompt = f"Here is the current page content:\n\n{existing_content}\n\nInstruction: {instruction}\n\nReturn ONLY the full updated content, nothing else."
        updated = await call_claude(edit_prompt)
        await update_notion_page_content(page_id, updated)
        return {"ai": "claude", "action": "edit", "page": page_name, "output": updated}

    # --- Normal routing ---
    ai_choice = route_prompt(prompt)
    context   = await get_knowledge_base_context()

    if ai_choice == "reddit":
        output = await call_reddit_search(prompt)
    elif ai_choice == "grok":
        output = await call_grok(prompt, context)
    elif ai_choice == "chatgpt":
        output = await call_chatgpt(prompt, context)
    else:
        output = await call_claude(prompt, context)

    # Write back to the Notion source row if triggered from Notion
    if source_page_id:
        await update_notion_row(source_page_id, output, ai_choice)

    return {"ai": ai_choice, "action": "response", "output": output}


# ─── Webhook Endpoints ────────────────────────────────────────────────────────

@app.post("/webhook/notion")
async def notion_webhook(request: Request, background_tasks: BackgroundTasks):
    """
    Called by Albato when a new row is added to your Notion Command Center DB.
    Albato should POST: { "page_id": "...", "prompt": "..." }
    """
    body = await request.json()
    page_id = body.get("page_id")
    prompt  = body.get("prompt", "").strip()

    if not prompt or not page_id:
        raise HTTPException(status_code=400, detail="Missing page_id or prompt")

    # Update status to Processing immediately
    await notion.pages.update(
        page_id=page_id,
        properties={"Status": {"select": {"name": "Processing"}}}
    )

    # Run in background so webhook returns fast
    background_tasks.add_task(dispatch, prompt, page_id)
    return {"status": "processing", "page_id": page_id}


@app.post("/webhook/slack")
async def slack_webhook(request: Request, background_tasks: BackgroundTasks):
    """
    Called by Albato when a message is posted to your Slack AI channel.
    Albato should POST: { "text": "...", "channel": "...", "user": "..." }
    """
    body = await request.json()

    # Handle Slack URL verification challenge
    if body.get("type") == "url_verification":
        return {"challenge": body.get("challenge")}

    text    = body.get("text", "").strip()
    channel = body.get("channel", "")
    user    = body.get("user", "")

    if not text:
        return {"status": "ignored"}

    async def process_and_respond():
        result = await dispatch(text)
        output = result.get("output", "No response.")
        ai     = result.get("ai", "AI")

        # Log to Notion
        await create_notion_page(
            title=f"Slack: {text[:60]}",
            content=output,
            ai_used=ai
        )

        # Reply back in Slack
        if SLACK_BOT_TOKEN and channel:
            async with httpx.AsyncClient() as client:
                await client.post(
                    "https://slack.com/api/chat.postMessage",
                    headers={"Authorization": f"Bearer {SLACK_BOT_TOKEN}"},
                    json={
                        "channel": channel,
                        "text": f"*[{ai.upper()}]*\n{output}"
                    }
                )

    background_tasks.add_task(process_and_respond)
    return {"status": "processing"}


@app.post("/webhook/discord")
async def discord_webhook(request: Request, background_tasks: BackgroundTasks):
    """
    Called by Albato when a message is posted to your Discord AI channel.
    Albato should POST: { "content": "...", "channel_id": "...", "webhook_url": "..." }
    """
    body = await request.json()
    text        = body.get("content", "").strip()
    webhook_url = body.get("webhook_url", "")

    if not text:
        return {"status": "ignored"}

    async def process_and_respond():
        result = await dispatch(text)
        output = result.get("output", "No response.")
        ai     = result.get("ai", "AI")

        # Log to Notion
        await create_notion_page(
            title=f"Discord: {text[:60]}",
            content=output,
            ai_used=ai
        )

        # Reply back via Discord webhook
        if webhook_url:
            async with httpx.AsyncClient() as client:
                await client.post(
                    webhook_url,
                    json={"content": f"**[{ai.upper()}]**\n{output}"}
                )

    background_tasks.add_task(process_and_respond)
    return {"status": "processing"}


@app.post("/prompt")
async def direct_prompt(request: Request):
    """
    Direct API call — useful for testing without Albato.
    POST: { "prompt": "your question here" }
    """
    body   = await request.json()
    prompt = body.get("prompt", "").strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Missing prompt")
    result = await dispatch(prompt)
    return result


@app.get("/health")
async def health():
    return {"status": "ok", "version": "1.0.0"}
