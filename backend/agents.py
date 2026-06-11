import os
import asyncio
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import ToolMessage
import kapruka_mcp

load_dotenv()

system_prompt = """You are a highly empathetic, witty, and warm Sri Lankan concierge and shopping assistant for Kapruka. You speak English, Sinhala, and Tanglish fluently.
CRITICAL INSTRUCTIONS FOR YOUR PERSONALITY:
- You are NOT a robotic search box. You are a human-like friend. 
- Have an opinion, add personality, and use Sri Lankan local flavour (e.g., 'Aiyo!', 'Machan', 'Shaa', 'Niyamai').
- Be proactive. Don't just list products; suggest a plan or a combination of items. 
- ALWAYS INCLUDE DETAILS: When suggesting items, always include the exact Product Name and Price so the user knows what they are buying.
- Never break character. Always be the warm, surprisingly helpful Sri Lankan concierge."""

prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"), 
])

async def chat(user_message: str):
    print("\n[System] Connecting to Kapruka and discovering tools...")
    session = await kapruka_mcp.get_client()
    
    mcp_tools_list = await session.list_tools()
    dynamic_tools = []
    
    # CORE TOOLS FILTER: Keep only what we need to save thousands of tokens
    allowed_tools = ["kapruka_search_products", "kapruka_get_product", "kapruka_list_categories"]
    
    for t in mcp_tools_list.tools:
        # Skip heavy tools we don't need right now to keep under the 8,000 token limit
        if t.name not in allowed_tools:
            continue
            
        flat_properties = {}
        flat_required = []
        
        if t.inputSchema and "properties" in t.inputSchema:
            if "params" in t.inputSchema["properties"]:
                params_obj = t.inputSchema["properties"]["params"]
                if isinstance(params_obj, dict) and "properties" in params_obj:
                    flat_properties = params_obj["properties"]
                    flat_required = params_obj.get("required", [])
            else:
                flat_properties = t.inputSchema.get("properties", {})
                flat_required = t.inputSchema.get("required", [])

        cleaned_properties = {}
        for key, value in flat_properties.items():
            if isinstance(value, dict):
                if "$ref" in value or any("$ref" in str(v) for v in value.values()):
                    continue
                # Token Saver: Shorten parameter descriptions if they are too wordy
                if "description" in value and len(value["description"]) > 100:
                    value["description"] = value["description"][:95] + "..."
                cleaned_properties[key] = value
            else:
                cleaned_properties[key] = value

        # Token Saver: Truncate tool description if it's massive
        tool_desc = t.description
        if tool_desc and len(tool_desc) > 150:
            tool_desc = tool_desc[:145] + "..."

        dynamic_tools.append({
            "type": "function",
            "function": {
                "name": t.name,
                "description": tool_desc,
                "parameters": {
                    "type": "object",
                    "properties": cleaned_properties,
                    "required": [r for r in flat_required if r in cleaned_properties]
                }
            }
        })
        
    print(f"✅ Dynamically loaded & token-optimized {len(dynamic_tools)} core tools!")

    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.7)
    llm_with_tools = llm.bind_tools(dynamic_tools)

    prompt_value = await prompt_template.ainvoke({
        "input": user_message,
        "agent_scratchpad": []
    })
    
    messages = prompt_value.to_messages()
    
    for step in range(5):
        response = await llm_with_tools.ainvoke(messages)
        
        if not response.tool_calls:
            return response.content
            
        print("🧠 AI is thinking... it needs to use a tool!")
        messages.append(response)
        
        for tool_call in response.tool_calls:
            print(f"🛠️  Executing Tool: {tool_call['name']}")
            
            args = tool_call["args"]
            
            if "product_id" in args and "id" not in args:
                args["id"] = args.pop("product_id")
            if "query" in args and "q" not in args:
                args["q"] = args.pop("query")
                
            mcp_args = {"params": args}
            
            raw_result = await kapruka_mcp.call_tool(tool_call["name"], mcp_args)
            messages.append(ToolMessage(content=str(raw_result), tool_call_id=tool_call["id"]))

if __name__ == "__main__":
    async def main():
        await chat("Ayubowan! list all the iphones available right now")
        
    asyncio.run(main())