# Interactive Sales Database Explorer

## Overview

This Python application provides an interactive interface for exploring and manipulating SQLite databases, particularly geared toward sales data. It allows users to:

*   **Upload databases:**  Import existing `.db` files or create new databases from CSV or Excel files.
*   **Browse table schemas:**  View available tables and their columns.
*   **Preview data:**  View sample data from selected tables.
*   **Filter and sort:**  Dynamically filter and sort data based on selected columns and values.
*   **Edit tables:** Modify table data directly.
*   **Execute pre-built queries:**  Run pre-defined queries for common sales analysis tasks (e.g., top-selling products, sales trends).
*   **Save/Load Sessions:** Save current states of filters, orderings, and selection states.

## Architecture

The application is built using Python and leverages the following libraries:

*   **sqlite3:**  For interacting with SQLite databases.
*   **pandas:**  For data manipulation and analysis.
*   **ipywidgets:**  For creating interactive UI elements in a Jupyter environment (like Google Colab).
*   **google.colab:** To access files and facilitate downloads in Google Colab
*   **chardet:** To detect encoding types.
*   **re:** for regular expressions
*   **io:** for memory-based I/O
*   **collections:** for defaultdict
*   **qgrid:** for interactive table editors
*   **json:** for saving and loading session data

The application follows a modular design, with distinct functions for:

*   **Database Interaction:** Connecting to databases, fetching schemas, loading data.
*   **UI Components:** Creating interactive widgets (dropdowns, sliders, checkboxes, etc.).
*   **Data Filtering & Sorting:** Dynamically filtering and sorting data based on user selections.
*   **Query Execution:**  Running pre-defined and user-defined queries.
*   **Session Management:** Saving and loading the application's state.

## Key Components

1.  **UI Widgets:**  The application's interface is built using a variety of ipywidgets, including:
    *   `ToggleButtons`: For selecting data sources and switching between editing modes.
    *   `SelectMultiple`: For choosing tables and filtering values.
    *   `Dropdown`: For selecting columns to filter and sort by.
    *   `FloatRangeSlider`:  For filtering numeric data.
    *   `Text`: for defining general value, for example SQL commands, for filtering
2.  **Data Filtering and Sorting:** The core filtering logic is applied as pandas queries, and is updated in real-time as user selections change.
3.  **Editable Tables:** The `qgrid` component enables direct modification of data within selected tables.
4.  **Pre-built Queries:** A selection of pre-defined queries are available for common sales analysis tasks.
5.  **Session Management:** The application's state can be saved to a JSON file, allowing users to resume their work later.

## Dependencies

To run this application, you'll need the following Python libraries installed:

```bash
pip install sqlite3 pandas ipywidgets google-colab chardet re io collections qgrid json
