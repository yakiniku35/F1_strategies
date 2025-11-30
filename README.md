# F1_strategies

## Troubleshooting

### ModuleNotFoundError: No module named 'plotly'

If you see this error when running the application:
```
ModuleNotFoundError: No module named 'plotly'
```

This means the required Python packages are not installed. The application depends on several packages (like `plotly`, `streamlit`, `fastf1`, etc.) listed in `requirements.txt`. You must install these dependencies before running the app.

**Solution:** Run `pip install -r requirements.txt` to install all required packages.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yakiniku35/F1_strategies.git
   cd F1_strategies
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up your environment variables by creating a `.env` file in the project root:
   ```
   GROQ_API_KEY=your_api_key_here
   ```

4. Run the application:
   ```bash
   streamlit run app.py
   ```