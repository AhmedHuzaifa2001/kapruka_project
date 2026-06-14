import json
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

async def chat_stream(user_message: str):
    print("\n[System] Connecting to Kapruka and discovering tools...")
    yield f"data: {json.dumps({'status': 'Connecting to Kapruka...'})}\n\n"
    
    session = await kapruka_mcp.get_client()
    mcp_tools_list = await session.list_tools()
    dynamic_tools = []
    
    allowed_tools = ["kapruka_search_products", "kapruka_get_product", "kapruka_list_categories"]
    
    for t in mcp_tools_list.tools:
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
                if "description" in value and len(value["description"]) > 100:
                    value["description"] = value["description"][:95] + "..."
                cleaned_properties[key] = value
            else:
                cleaned_properties[key] = value

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
            async for chunk in llm.astream(messages):
                if chunk.content:
                    yield f"data: {json.dumps({'chunk': chunk.content})}\n\n"
            break 
            
        messages.append(response)
        
        for tool_call in response.tool_calls:
            tool_name = tool_call['name']
            print(f"🛠️  Executing Tool: {tool_name}")
            
           
            yield f"data: {json.dumps({'status': f'Using tool: {tool_name} ...'})}\n\n"
            
            args = tool_call["args"]
            
            if "product_id" in args and "id" not in args:
                args["id"] = args.pop("product_id")
            if "query" in args and "q" not in args:
                args["q"] = args.pop("query")
                
            mcp_args = {"params": args}
            
            raw_result = await kapruka_mcp.call_tool(tool_name, mcp_args)
            messages.append(ToolMessage(content=str(raw_result), tool_call_id=tool_call["id"]))

