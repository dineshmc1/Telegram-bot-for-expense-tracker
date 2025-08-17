import os
import io
import base64
import requests
import json
import gspread
import logging
from typing import TypedDict, List, Union

from dotenv import load_dotenv
from telegram import Update, Bot
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from google.oauth2.service_account import Credentials
from google.auth import default as google_auth_default

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END, START

load_dotenv()

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GOOGLE_SHEET_URL = "https://docs.google.com/spreadsheets/d/1_cXQu4npDWdowuYIuFLOI6jPAvW5K6kWhyVk3vw-SsY/edit?usp=sharing" 
GOOGLE_SHEET_NAME = "Sheet1" 
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

SYSTEM_PROMPT = (
    "You are a personal finance assistant. Your task is to help the user "
    "track their expenses. When you receive a receipt image, "
    "your task is to extract the following information: store name, "
    "purchase date (in 'YYYY-MM-DD' format), a list of items and their prices, "
    "and a general category (e.g., 'Groceries', 'Restaurant', 'Gas'). "
    "After extracting the information, use the 'save_expense_to_sheet' tool to "
    "record each item as a separate expense entry. For example, if a receipt "
    "has milk for $4 and bread for $3, call the tool twice. If the user asks for "
    "expenses on a specific date, use the 'get_expenses_by_date' tool. "
    "If they are just chatting, respond conversationally."
)


def setup_google_sheets_client():
    try:
        service_account_path = 'service_account.json'
        if not os.path.exists(service_account_path):
            logger.error("service_account.json file not found.")
            raise FileNotFoundError("Service account key file not found.")
        
        gc = gspread.service_account(filename=service_account_path)
        spreadsheet = gc.open_by_url(GOOGLE_SHEET_URL)
        return spreadsheet.worksheet(GOOGLE_SHEET_NAME)
    except Exception as e:
        logger.error(f"Error setting up Google Sheets client: {e}")
        return None

class Expense(BaseModel):
    item: str = Field(description="The name of the purchased item.")
    price: float = Field(description="The price of the item.")
    date: str = Field(description="The date of the purchase in 'YYYY-MM-DD' format.")
    store: str = Field(description="The name of the store where the item was purchased.")
    category: str = Field(description="The category of the expense (e.g., 'Groceries', 'Utilities', 'Restaurants').")

class ExpenseTools:
    def __init__(self):
        self.sheet = setup_google_sheets_client()
        if self.sheet:
            if not self.sheet.row_values(1):
                self.sheet.append_row(['Date', 'Store', 'Item', 'Price', 'Category'])

    def save_expense_to_sheet(self, expense: Expense):
        if not self.sheet:
            return "Google Sheets client is not initialized."
        try:
            self.sheet.append_row([expense.date, expense.store, expense.item, expense.price, expense.category])
            return f"Successfully saved expense: {expense.item} for ${expense.price}."
        except Exception as e:
            logger.error(f"Error saving expense to sheet: {e}")
            return f"Failed to save expense due to an error."

    def get_expenses_by_date(self, date: str):
        if not self.sheet:
            return "Google Sheets client is not initialized."
        try:
            records = self.sheet.get_all_records()
            expenses = [record for record in records if record['Date'] == date]
            if not expenses:
                return f"No expenses found for {date}."
            
            total_expense = sum(float(e['Price']) for e in expenses)
            summary = f"Expenses for {date}:\n"
            for exp in expenses:
                summary += f"- {exp['Item']} at {exp['Store']}: ${exp['Price']} ({exp['Category']})\n"
            summary += f"Total expense on {date}: ${total_expense:.2f}"
            return summary
        except Exception as e:
            logger.error(f"Error getting expenses: {e}")
            return f"Failed to retrieve expenses due to an error."

tools = ExpenseTools()
tool_list = [tools.save_expense_to_sheet, tools.get_expenses_by_date]

class AgentState(TypedDict):
    """The state of our Langgraph agent."""
    messages: List[Union[HumanMessage, AIMessage, ToolMessage]]
    image_data: bytes | None

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=OPENAI_API_KEY)
llm_with_tools = llm.bind_tools(tool_list)

def handle_message_node(state: AgentState):
    messages = state["messages"]
    image_data = state.get("image_data")
    
    if image_data:
        try:
            logger.info("Image data found, providing to LLM for analysis.")
            base64_image = base64.b64encode(image_data).decode('utf-8')
            image_part = {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
            }
            if messages[-1].content and isinstance(messages[-1].content, str):
                messages[-1].content = [
                    {"type": "text", "text": messages[-1].content},
                    image_part
                ]
            else:
                 messages[-1].content = [image_part]

        except Exception as e:
            logger.error(f"Error processing image: {e}")
            messages[-1] = HumanMessage(content="I am having trouble processing this image. Please try again.")

    response = llm_with_tools.invoke(messages)
    state["messages"].append(response)
    state["image_data"] = None 
    return state

def tool_node(state: AgentState):
    last_message = state["messages"][-1]
    
    tool_calls = last_message.tool_calls
    
    for tool_call in tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_call_id = tool_call["id"]
        
        if hasattr(tools, tool_name):
            tool_func = getattr(tools, tool_name)
            try:
                if tool_name == 'save_expense_to_sheet' and 'expense' in tool_args:
                    expense_data = tool_args['expense']
                    tool_output = tool_func(expense=Expense(**expense_data))
                else:
                    tool_output = tool_func(**tool_args)
                state["messages"].append(ToolMessage(content=tool_output, tool_call_id=tool_call_id))
            except Exception as e:
                error_message = f"Error executing tool '{tool_name}': {e}"
                state["messages"].append(ToolMessage(content=error_message, tool_call_id=tool_call_id))
        else:
            error_message = f"Tool '{tool_name}' not found."
            state["messages"].append(ToolMessage(content=error_message, tool_call_id=tool_call_id))
            
    return state

def should_continue(state: AgentState):
    last_message = state["messages"][-1]
    if not last_message.tool_calls:
        return "end"
    else:
        return "continue"

graph = StateGraph(AgentState)
graph.add_node("handle_message", handle_message_node)
graph.add_node("tool_node", tool_node)
graph.add_edge(START, "handle_message")
graph.add_conditional_edges("handle_message", should_continue, {
    "continue": "tool_node",
    "end": END
})
graph.add_edge("tool_node", "handle_message")
app = graph.compile()


user_states = {}

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Hello! I'm your personal finance tracker bot. Send me a receipt, and I'll track your expenses. You can also ask me for your expenses on a specific date.")
    user_id = update.effective_user.id
    user_states[user_id] = AgentState(messages=[SystemMessage(content=SYSTEM_PROMPT)], image_data=None)
    logger.info(f"User {user_id} started a new conversation.")

async def handle_text_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    user_text = update.message.text
    
    if user_id not in user_states:
        user_states[user_id] = AgentState(messages=[SystemMessage(content=SYSTEM_PROMPT)], image_data=None)
        
    conversation_history = user_states[user_id]["messages"]
    conversation_history.append(HumanMessage(content=user_text))
    
    result = app.invoke({'messages': conversation_history, 'image_data': None})
    
    last_agent_message = result["messages"][-1]
    if last_agent_message.content:
        await update.message.reply_text(last_agent_message.content)
    
    user_states[user_id] = result
    logger.info(f"User {user_id} message handled.")

async def handle_photo_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    
    if user_id not in user_states:
        user_states[user_id] = AgentState(messages=[SystemMessage(content=SYSTEM_PROMPT)], image_data=None)
    
    file_id = update.message.photo[-1].file_id
    file = await context.bot.get_file(file_id)
    
    image_stream = io.BytesIO()
    await file.download_to_memory(out=image_stream)
    image_stream.seek(0)
    
    image_data = image_stream.read()
    
    await update.message.reply_text("Thanks! I'm analyzing your receipt now...")
    
    conversation_history = user_states[user_id]["messages"]
    conversation_history.append(HumanMessage(content="Please analyze this receipt."))
    
    result = app.invoke({'messages': conversation_history, 'image_data': image_data})
    
    last_agent_message = result["messages"][-1]
    if last_agent_message.content:
        await update.message.reply_text(last_agent_message.content)
    
    user_states[user_id] = result
    logger.info(f"User {user_id} photo handled.")

def main():
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    
    application.add_handler(CommandHandler("hello", start))
    application.add_handler(MessageHandler(filters.PHOTO, handle_photo_message))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text_message))
    
    logger.info("Bot is starting...")
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
