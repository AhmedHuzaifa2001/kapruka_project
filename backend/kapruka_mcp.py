import asyncio
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

_global_client = None
_lock = asyncio.Lock()  

# The Background Worker (Stays in the room holding the phone)
async def _background_connector():
    global _global_client
    server_url = "https://mcp.kapruka.com/mcp"
    
    # Safe context managers (no AsyncExitStack hacks)
    async with streamable_http_client(server_url) as streams:
        read_stream, write_stream = streams[0], streams[1]
        
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            _global_client = session
            
            # This line keeps the worker alive forever so anyio doesn't crash
            await asyncio.Event().wait() 

# 2. The Bouncer (Handles the traffic)
async def get_client():
    global _global_client

    async with _lock:  # lock prevents race conditions
        if _global_client is None:
            # Tell the worker to start dialing
            asyncio.create_task(_background_connector())
            
            # Wait until the worker successfully connects
            while _global_client is None:
                await asyncio.sleep(0.1)
                
    return _global_client

# 3. The Caller
async def call_tool(name, args):
    session = await get_client()
    result = await session.call_tool(name, arguments=args)
    texts = [block.text for block in result.content if hasattr(block, "text")]
    return "\n".join(texts)

if __name__ == "__main__":
    async def test():
        print("Starting test...")
        result = await call_tool("kapruka_list_categories", {"params": {}})
        print("\n--- RESULTS ---")
        print(result)
        
    asyncio.run(test())