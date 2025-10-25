import discord
import asyncio
from dotenv import load_dotenv
import os
from time import sleep

class DiscordBot:
    def __init__(self, token):
        self.token = token
        self.intents = discord.Intents.default()
        self.intents.message_content = True
        self.client = discord.Client(intents=self.intents)
        
        
    async def get_recent_messages(self, channel_id, limit=5):
        """Get the most recent messages from a channel"""
        channel = self.client.get_channel(channel_id)
        if channel:
            messages = []
            async for message in channel.history(limit=limit):
                messages.append({
                    'author': str(message.author),
                    'content': message.content,
                    'timestamp': message.created_at
                })
            return messages[::-1]
        else:
            print("ERR: Channel didn't register.")
            return []
        
        
    async def send_message(self, channel_id, content):
        """Send a msg to a channel"""
        try:
            channel = self.client.get_channel(channel_id)
            if channel and hasattr(channel, 'send'):
                await channel.send(content)
                return True # Success!
            else:
                print(f"send_message: count not send message to {channel}")
                return False
        except discord.Forbidden:
            print("send_message: Bot doesn't have permission to send messages in this channel")
            return False
        except discord.HTTPException as e:
            print(f"send_message: Failed to send message: {e}")
            return False
        except Exception as e:
            print(f"send_message: encountered error {e}")
            return False
    
    
    
    async def run(self, debug=False, use_content=False):
        @self.client.event
        async def on_ready(): # Discord event - triggered when bot is logged in and ready
            print(f'Bot {self.client.user} is ready!')
            print(f"Bot logged in as: {self.client.user}")
            print(f"Bot is in {len(self.client.guilds)} server(s):")
            if debug:
                for guild in self.client.guilds:
                    print(f"\nServer: {guild.name}")
                    print(f"    Bot permissions: {guild.me.guild_permissions}")
                    
                    text_channels = [ch for ch in guild.text_channels if ch.permissions_for(guild.me).read_messages]
                    print(f"    Accessible text channels: {len(text_channels)}")
            
                for channel in text_channels[:3]:  # Show first 3 channels
                    perms = channel.permissions_for(guild.me)
                    print(f"    - #{channel.name}: id={channel.id} read_messages={perms.read_messages}, read_message_history={perms.read_message_history}")
            
            print("Setting up the target channel...")
            target_guild = self.client.guilds[0]
            text_channels = target_guild.text_channels
            
            # Find matching channel using a generator
            target_channel = next((channel for channel in text_channels if channel.name == 'duck-hunt'), None)
            if target_channel:
                print("Found channel!")
            else:
                print("Could not find channel...")
            channel_id = target_channel.id
            print(f"Channel: {target_channel.name}, ID: {channel_id}, type: {type(channel_id)}")
                                    
            assert channel_id, "Channel ID is none? What???"
            while True:
                # print("Reading most recent message...")
                msg = (await self.get_recent_messages(channel_id, 1))[0]
                
                if debug: 
                    print(f"[{msg['timestamp']}] {msg['author']}: {msg['content']}") 
                    key = 'author' if not use_content else 'content' 
                    if 'duckhunt' not in msg[key].lower():
                        if 'duck' in msg[key].lower():
                            await self.send_message(channel_id, "!pew")
                            print("Got a duck!") 
                        elif 'dead' in msg[key].lower():
                            await self.send_message(channel_id, "!revive") 
                            print("Revived!") 
                        elif 'jam' in msg[key].lower() or '0/6' in msg[key].lower():
                            await self.send_message(channel_id, "!reload") 
                            print("Reloaded!") 
                else:
                    if 'duckhunt' not in msg['author'].lower():
                        if 'duck' in msg['author'].lower():
                            await self.send_message(channel_id, "!pew")
                            print("Got a duck!") 
                        elif 'dead' in msg['content'].lower():
                            await self.send_message(channel_id, "!revive") 
                            print("Revived!") 
                        elif 'jam' in msg['content'].lower() or '0/6' in msg[key].lower():
                            await self.send_message(channel_id, "!reload") 
                            print("Reloaded!") 
                sleep(1) # Check every second to avoid triggering rate limit


        await self.client.start(self.token)
        

async def main():
    load_dotenv()
    BOT_TOKEN = os.getenv("BOT_TOKEN")
    bot = DiscordBot(BOT_TOKEN)
    await bot.run(use_content=False)

if __name__ == "__main__":
    asyncio.run(main())