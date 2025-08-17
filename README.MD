# Telegram Receipt Bot - Personal Finance Tracker

A smart Telegram bot that helps you track your expenses by analyzing receipt images using AI. The bot extracts purchase information from receipts and automatically saves it to a Google Sheets spreadsheet for easy expense tracking and analysis.

## Features

- **Receipt Image Analysis**: Upload receipt photos and the bot will automatically extract:
  - Store name
  - Purchase date
  - Individual items and their prices
  - Expense category (Groceries, Restaurant, Gas, etc.)

- **Automatic Expense Tracking**: Each item from the receipt is saved as a separate expense entry in Google Sheets

- **Expense Queries**: Ask the bot about your expenses on specific dates

- **Conversational AI**: Powered by OpenAI's GPT-4o-mini for natural language processing

- **Google Sheets Integration**: All expenses are automatically organized in a Google Sheets spreadsheet

## Prerequisites

Before running this bot, you'll need:

1. **Python 3.8+** installed on your system
2. **Telegram Bot Token** - Create a bot via [@BotFather](https://t.me/botfather)
3. **OpenAI API Key** - Get from [OpenAI Platform](https://platform.openai.com/)
4. **Google Sheets API** - Set up a Google Cloud project and enable Google Sheets API
5. **Service Account Key** - Download the `service_account.json` file from Google Cloud Console

## Installation

1. **Clone or download this repository**

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:
   Create a `.env` file in the project root with:
   ```
   TELEGRAM_BOT_TOKEN=your_telegram_bot_token_here
   OPENAI_API_KEY=your_openai_api_key_here
   ```

4. **Configure Google Sheets**:
   - Place your `service_account.json` file in the project root
   - Update the `GOOGLE_SHEET_URL` in `bot.py` with your Google Sheets URL
   - Make sure the service account has edit permissions on the spreadsheet

## Usage

### Starting the Bot

Run the bot with:
```bash
python bot.py
```

### Bot Commands

- **Start**: Send `/hello` to start the bot and initialize your session

### How to Use

1. **Send a receipt photo**: Simply take a photo of your receipt and send it to the bot
2. **Ask about expenses**: Type queries like "Show me expenses for 2024-01-15"
3. **Chat naturally**: The bot can handle conversational requests about your finances

### Example Interactions

**User**: *sends receipt photo*
**Bot**: "Thanks! I'm analyzing your receipt now..."
**Bot**: "Successfully saved expense: Milk for $4.50. Successfully saved expense: Bread for $3.25."

**User**: "Show me expenses for 2024-01-15"
**Bot**: "Expenses for 2024-01-15:
- Milk at Walmart: $4.50 (Groceries)
- Bread at Walmart: $3.25 (Groceries)
Total expense on 2024-01-15: $7.75"

## Project Structure

```
telegram bot reciept/
├── bot.py                 # Main bot application
├── requirements.txt       # Python dependencies
├── service_account.json   # Google Sheets API credentials
├── .env                   # Environment variables (create this)
├── expenseTracker/        # Virtual environment
└── README.md             # This file
```

## Technical Details

### Architecture

The bot uses **LangGraph** for workflow management and **LangChain** for AI interactions:

- **StateGraph**: Manages conversation flow and tool execution
- **OpenAI GPT-4o-mini**: Processes images and text for expense extraction
- **Google Sheets API**: Stores and retrieves expense data
- **Telegram Bot API**: Handles user interactions

### Key Components

- **ExpenseTools**: Handles Google Sheets operations
- **AgentState**: Manages conversation state and image data
- **Tool Integration**: Seamlessly connects AI analysis with data storage

## Configuration

### Google Sheets Setup

1. Create a new Google Sheets spreadsheet
2. Share it with your service account email (found in `service_account.json`)
3. Update the `GOOGLE_SHEET_URL` variable in `bot.py`
4. The bot will automatically create headers: `Date`, `Store`, `Item`, `Price`, `Category`

### Environment Variables

Required environment variables in `.env`:
- `TELEGRAM_BOT_TOKEN`: Your Telegram bot token
- `OPENAI_API_KEY`: Your OpenAI API key

## Troubleshooting

### Common Issues

1. **"service_account.json file not found"**
   - Ensure the file is in the project root directory
   - Check file permissions

2. **"Google Sheets client is not initialized"**
   - Verify your service account has edit permissions
   - Check the Google Sheets URL is correct

3. **"Failed to save expense"**
   - Ensure the spreadsheet is accessible
   - Check internet connectivity

### Logging

The bot includes comprehensive logging. Check the console output for detailed error messages and debugging information.

## Contributing

Feel free to submit issues and enhancement requests!

## License

This project is open source and available under the [MIT License](LICENSE).

## Support

If you encounter any issues or have questions, please check the troubleshooting section above or create an issue in the repository.