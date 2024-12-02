# src/discord_bot/discord_bot.py

import discord
import asyncio
import logging
from discord.ext import commands, tasks
from src.config import Config
from src.ai_generator import AIGenerator
from src.memory_processor import process_daily_memories
from src.memory_decision import select_relevant_memories
from src.story_circle_manager import get_current_context, update_story_circle, progress_narrative
from datetime import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('DiscordBot')

class DiscordBot(commands.Bot):
    def __init__(self):
        intents = discord.Intents.default()
        intents.message_content = True  # Enable message content intent
        super().__init__(command_prefix='!', intents=intents)
        self.generator = AIGenerator(mode='discord')
        self.user_conversations = {}
        self.MAX_MEMORY = Config.MAX_MEMORY
        self.remove_command('help')  # Remove default help command if desired

    async def on_ready(self):
        logger.info(f'Logged in as {self.user.name} (ID: {self.user.id})')
        self.process_memories.start()
        self.update_narrative.start()
        print('Discord AI Bot is online!')

    async def on_message(self, message):
        if message.author == self.user:
            return  # Ignore messages from the bot itself

        if self.user.mentioned_in(message):
            await self.handle_mention(message)

        await self.process_commands(message)  # Process other commands if any

    async def handle_mention(self, message):
        try:
            logger.info(f'Mention received from {message.author}: {message.content}')
            user_id = message.author.id
            username = message.author.name
            user_message = message.content.replace(f'<@{self.user.id}>', '').strip()

            # Get all necessary context
            conversation_context = self.get_conversation_context(user_id)
            memories = await select_relevant_memories(username, user_message)
            narrative_context = get_current_context()

            # Generate response with full context
            response = await self.generator.generate_content(
                user_message=user_message,
                user_id=user_id,
                username=username,
                conversation_context=conversation_context,
                memories=memories,
                narrative_context=narrative_context
            )

            # Send the response
            await message.channel.send(response)
            logger.info('Response sent.')

            # Update conversation history
            self.add_to_conversation_history(user_id, user_message, is_bot=False)
            self.add_to_conversation_history(user_id, response, is_bot=True)

        except Exception as e:
            logger.error(f'Error handling mention: {e}')
            await message.channel.send("Sorry, I couldn't process your request at the moment.")

    def add_to_conversation_history(self, user_id, message, is_bot):
        if user_id not in self.user_conversations:
            self.user_conversations[user_id] = []
        self.user_conversations[user_id].append({
            'content': message,
            'is_bot': is_bot,
            'timestamp': asyncio.get_event_loop().time()
        })
        # Keep only the last MAX_MEMORY messages
        if len(self.user_conversations[user_id]) > self.MAX_MEMORY:
            self.user_conversations[user_id].pop(0)

    def get_conversation_context(self, user_id):
        """Returns conversation history in a structured format"""
        history = self.user_conversations.get(user_id, [])
        return [
            {
                "role": "assistant" if msg['is_bot'] else "user",
                "content": msg['content']
            }
            for msg in history
        ]

    @commands.command(name='chatid')
    async def chatid(self, ctx):
        """Returns the chat ID."""
        chat_id = ctx.guild.id if ctx.guild else ctx.author.id
        await ctx.send(f'Chat ID: {chat_id}')

    @tasks.loop(time=time(hour=23, minute=55))  # Run at 23:55 every day
    async def process_memories(self):
        try:
            logger.info("Starting nightly memory processing...")
            await process_daily_memories(self.user_conversations)
            # Clear the day's conversations after processing
            self.user_conversations.clear()
            logger.info("Nightly memory processing completed")
        except Exception as e:
            logger.error(f"Error in nightly memory processing: {e}")

    @tasks.loop(hours=6)
    async def update_narrative(self):
        try:
            logger.info("Progressing story circle narrative...")
            await progress_narrative()  # This will either move to next event or generate new content
            logger.info("Story circle progression completed")
        except Exception as e:
            logger.error(f"Error in story circle progression: {e}")

    async def on_error(self, event, *args, **kwargs):
        logger.error(f'Error in event {event}: {args} {kwargs}')
        if event == 'on_message':
            await args[0].channel.send("An unexpected error occurred while processing your message.")

    def run_bot(self):
        self.run(Config.DISCORD_BOT_TOKEN)