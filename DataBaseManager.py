import sqlite3
import pandas as pd
import ipywidgets as widgets
from IPython.display import display, clear_output
import os
from io import BytesIO
from google.colab import files
import json
import chardet
import re
import io
from collections import defaultdict
import qgrid
import ipywidgets as widgets

# ===== Helpers =====
def clean_column_name(col):
    """Standardize column names for SQLite compatibility."""
    col = re.sub(r'[^a-zA-Z0-9_ ]', '', str(col)).replace(' ', '_').lower()
    col = re.sub(r'_{2,}', '_', col).strip('_')
    return col if col else 'unnamed_column'

def fetch_schema():
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [t[0] for t in cur.fetchall()]
    columns = {}
    for table in tables:
        cur.execute(f'PRAGMA table_info("{table}")')
        columns[table] = [col[1] for col in cur.fetchall()]
    return tables, columns

def save_dataframe_to_db(df, table_name):
    df.columns = [clean_column_name(c) for c in df.columns]
    df.to_sql(table_name, conn, if_exists='replace', index=False)

def refresh_schema():
    global tables, columns
    tables, columns = fetch_schema()
    table_selector.options = tables
    preview_selected_tables({'new': table_selector.value})

def preview_selected_tables(change):
    preview_box.clear_output()
    with preview_box:
        for table in table_selector.value:
            df = pd.read_sql_query(f"SELECT * FROM {table} LIMIT 5", conn)
            display(widgets.HTML(f"<b>Preview of {table}:</b>"))
            display(df)
    update_filter_widgets()

# --- Filters code ---
selected_filter_values = defaultdict(set)

def create_categorical_filter(col, options):
    combo = widgets.Combobox(
        placeholder=f"Type or select {col}",
        options=sorted(options),
        description=col,
        ensure_option=True,
        continuous_update=False,
        layout=widgets.Layout(width='280px')
    )

    add_btn = widgets.Button(description="Add", button_style='success', layout=widgets.Layout(width='60px'))
    remove_btn = widgets.Button(description="Remove", button_style='danger', layout=widgets.Layout(width='80px'))

    selected_box = widgets.SelectMultiple(
        options=[],
        description='Selected',
        layout=widgets.Layout(width='300px', height='80px')
    )

    def add_value(_):
        val = combo.value
        if val and val not in selected_filter_values[col]:
            selected_filter_values[col].add(val)
            selected_box.options = sorted(selected_filter_values[col])
        combo.value = ''

    def remove_selected(_):
        for val in selected_box.value:
            selected_filter_values[col].discard(val)
        selected_box.options = sorted(selected_filter_values[col])

    add_btn.on_click(add_value)
    remove_btn.on_click(remove_selected)

    widget_box = widgets.VBox([
        widgets.HBox([combo, add_btn, remove_btn]),
        selected_box
    ])
    return widget_box

def update_filter_widgets():
    filter_widgets = []
    if not table_selector.value:
        filter_box.children = []
        orderby_dropdown.options = ['']
        groupby_dropdown.options = ['']
        return

    all_cols = sorted(set(sum([columns[t] for t in table_selector.value], [])))

    for col in list(selected_filter_values.keys()):
        if col not in all_cols:
            del selected_filter_values[col]

    for col in all_cols:
        try:
            sample_query = f"SELECT {col} FROM {table_selector.value[0]} WHERE {col} IS NOT NULL LIMIT 100"
            sample_vals = pd.read_sql_query(sample_query, conn)[col].dropna().unique()
            if pd.api.types.is_numeric_dtype(sample_vals):
                slider = widgets.FloatRangeSlider(
                    description=col,
                    min=float(pd.Series(sample_vals).min()),
                    max=float(pd.Series(sample_vals).max()),
                    step=0.01,
                    continuous_update=False,
                    layout=widgets.Layout(width='90%')
                )
                filter_widgets.append(slider)
            else:
                cat_filter = create_categorical_filter(col, sample_vals)
                filter_widgets.append(cat_filter)
        except Exception as e:
            filter_widgets.append(widgets.Text(description=col, placeholder="e.g., = 'Milk'"))

    filter_box.children = filter_widgets
    orderby_dropdown.options = [''] + all_cols
    groupby_dropdown.options = [''] + all_cols

def run_query(_):
    if not table_selector.value:
        output_box.clear_output()
        with output_box:
            print("\u26A0\ufe0f No tables selected.")
        return

    selected_tables = table_selector.value
    all_cols = sorted(set(sum([columns[t] for t in selected_tables], [])))
    select_cols = ', '.join([f'{t}.{c}' for t in selected_tables for c in columns[t]])
    from_clause = ', '.join(selected_tables)

    where_clauses = []

    for w in filter_box.children:
        if isinstance(w, widgets.FloatRangeSlider):
            col = w.description
            if w.value[0] != w.min or w.value[1] != w.max:
                where_clauses.append(f"{col} BETWEEN {w.value[0]} AND {w.value[1]}")

        elif isinstance(w, widgets.VBox):
            cat_filter_box = w
            col = cat_filter_box.children[0].children[0].description
            selected_vals = cat_filter_box.children[1].options
            if selected_vals:
                quoted_vals = ", ".join([f"'{val}'" for val in selected_vals])
                where_clauses.append(f"{col} IN ({quoted_vals})")

        elif isinstance(w, widgets.Text):
            col = w.description
            val = w.value.strip()
            if val:
                if not re.match(r"^[<>=!]", val):
                    val = f"= '{val}'"
                where_clauses.append(f"{col} {val}")

    where_clause = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ''
    order_clause = f"ORDER BY {orderby_dropdown.value}" if orderby_dropdown.value else ''
    group_clause = f"GROUP BY {groupby_dropdown.value}" if groupby_dropdown.value else ''

    query = f"SELECT {select_cols} FROM {from_clause} {where_clause} {group_clause} {order_clause} LIMIT 100;"
    output_box.clear_output()
    with output_box:
        try:
            df = pd.read_sql_query(query, conn)
            output_box.df_result = df
            display(widgets.HTML(f"<b>Query Result ({len(df)} rows):</b>"))
            display(df)
        except Exception as e:
            print("\u274C Query failed:", e)

def export_to_csv(_):
    if not hasattr(output_box, 'df_result'):
        with output_box:
            print("⚠️ No data to export.")
        return
    df = output_box.df_result
    temp_path = "/content/query_result.csv"
    df.to_csv(temp_path, index=False)
    files.download(temp_path)


def save_config(_):
    config_filters = {}
    for w in filter_box.children:
        if hasattr(w, 'description'):
            # For normal widgets like Text, Dropdown, Slider
            config_filters[w.description] = getattr(w, 'value', None)
        elif isinstance(w, widgets.VBox):
            # For VBox (categorical filters)
            # The column name is stored in the description of the Combobox inside first child of VBox
            combo = w.children[0].children[0]  # HBox -> Combobox is first child
            col = combo.description
            # The selected values are in SelectMultiple (second child of VBox)
            selected_vals = w.children[1].options
            config_filters[col] = list(selected_vals)
    config = {
        'tables': list(table_selector.value),
        'filters': config_filters,
        'order_by': orderby_dropdown.value,
        'group_by': groupby_dropdown.value
    }
    with open("session_config.json", "w") as f:
        json.dump(config, f)
    files.download("session_config.json")


def load_config(_):
    uploaded = files.upload()
    fname = next(iter(uploaded))
    with open(fname, 'r') as f:
        config = json.load(f)
    table_selector.value = tuple(config['tables'])
    preview_selected_tables(None)

    for w in filter_box.children:
        if hasattr(w, 'description') and w.description in config['filters']:
            # Normal widgets
            w.value = config['filters'][w.description]
        elif isinstance(w, widgets.VBox):
            combo = w.children[0].children[0]
            col = combo.description
            if col in config['filters']:
                selected_vals = config['filters'][col]
                # Set SelectMultiple options
                w.children[1].options = selected_vals
    orderby_dropdown.value = config['order_by']
    groupby_dropdown.value = config['group_by']


def run_prebuilt(change):
    val = change['new']
    if val == 'Top 10 SKUs by Value':
        query = "SELECT `SKU Code`, SUM(Value) as TotalValue FROM sales GROUP BY `SKU Code` ORDER BY TotalValue DESC LIMIT 10"
    elif val == 'Sales Trend by Month':
        query = "SELECT Month, SUM(Value) as TotalSales FROM sales GROUP BY Month ORDER BY Month"
    else:
        return
    output_box.clear_output()
    with output_box:
        try:
            df = pd.read_sql_query(query, conn)
            output_box.df_result = df
            display(widgets.HTML(f"<b>{val}:</b>"))
            display(df)
        except Exception as e:
            print("\u274C Query failed:", e)


# ===== Upload DB or Excel/CSV =====
upload_type = widgets.ToggleButtons(
    options=['📁 Upload .db file', '📊 Upload Excel/CSV'],
    description='Data Source:',
    button_style='info'
)

confirm_delete_checkbox = widgets.Checkbox(
    value=False,
    description="Yes, I really want to delete these tables.",
    layout=widgets.Layout(width='100%')
)

upload_button = widgets.Button(description='Upload', button_style='primary')
upload_output = widgets.Output()

def create_db_from_excel(temp_db_path):
    uploaded = files.upload()
    if not uploaded:
        print("⚠️ No files uploaded.")
        return None

    conn = sqlite3.connect(temp_db_path)

    for fname, content in uploaded.items():
        ext = fname.split('.')[-1].lower()
        base_name = re.sub(r'\W+', '_', fname.split('.')[0]).lower()
        try:
            if ext == 'csv':
                detected = chardet.detect(content)
                df = pd.read_csv(io.BytesIO(content), encoding=detected['encoding'])
                df.columns = [clean_column_name(c) for c in df.columns]
                df.to_sql(base_name, conn, if_exists='replace', index=False)
                print(f"✅ CSV '{fname}' loaded as table '{base_name}'")

            elif ext in ['xlsx', 'xls']:
                xl = pd.ExcelFile(io.BytesIO(content))
                for sheet in xl.sheet_names:
                    df = xl.parse(sheet)
                    if df.empty:
                        continue
                    sheet_name = re.sub(r'\W+', '_', sheet.lower())
                    df.columns = [clean_column_name(c) for c in df.columns]
                    table_name = f"{base_name}_{sheet_name}"
                    df.to_sql(table_name, conn, if_exists='replace', index=False)
                    print(f"✅ Sheet '{sheet}' loaded as table '{table_name}'")

            else:
                print(f"⚠️ Unsupported file type: {fname}")
        except Exception as e:
            print(f"❌ Error loading '{fname}': {e}")

    conn.commit()
    conn.close()
    return temp_db_path

def upload_existing_db():
    print("📁 Upload your `.db` file")
    uploaded = files.upload()
    if not uploaded:
        print("❌ No file uploaded")
        return None
    fname = next(iter(uploaded))
    path = f"/content/{fname}"
    print(f"✅ Uploaded: {fname}")
    return path

def upload_handler(_):
    global conn, DB_PATH, tables, columns

    upload_output.clear_output()
    with upload_output:
        if upload_type.value == '📁 Upload .db file':
            DB_PATH = upload_existing_db()
            if DB_PATH is None:
                print("❌ No DB uploaded.")
                return
            try:
                conn = sqlite3.connect(DB_PATH)
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                tables = [r[0] for r in cursor.fetchall()]
                print("📊 Tables found:", tables)
            except Exception as e:
                print("❌ Failed to open DB:", e)
                return

        elif upload_type.value == '📊 Upload Excel/CSV':
            temp_db_path = "/content/auto_generated.db"
            DB_PATH = create_db_from_excel(temp_db_path)
            if DB_PATH is None:
                print("❌ Failed to create DB from Excel/CSV.")
                return
            conn = sqlite3.connect(DB_PATH)
            print(f"✅ Created new DB from Excel/CSV: {DB_PATH}")

        tables, columns = fetch_schema()
        table_selector.options = tables
        preview_selected_tables({'new': table_selector.value})

upload_button.on_click(upload_handler)

# ===== Widgets for the main UI =====
table_selector = widgets.SelectMultiple(options=[], description='Tables:', layout=widgets.Layout(width='50%'))
filter_box = widgets.VBox()
orderby_dropdown = widgets.Dropdown(description='Order by:')
groupby_dropdown = widgets.Dropdown(description='Group by:')
export_button = widgets.Button(description='Export CSV', button_style='info')
save_config_button = widgets.Button(description='Save Session', button_style='warning')
load_config_button = widgets.Button(description='Load Session')
run_button = widgets.Button(description='\U0001F50E Run Query', button_style='success')
prebuilt_dropdown = widgets.Dropdown(description='Prebuilt:', options=['', 'Top 10 SKUs by Value', 'Sales Trend by Month'])
output_box = widgets.Output()
preview_box = widgets.Output()
edit_output = widgets.Output()

# === Filter & Sort widgets for 📊 Interactive Sales DB Explorer ===
main_filter_column = widgets.Dropdown(description="Filter by:", layout=widgets.Layout(width="200px"))
main_filter_value = widgets.Dropdown(description="Value:", layout=widgets.Layout(width="200px"))
main_sort_by = widgets.Dropdown(description="Sort by:", layout=widgets.Layout(width="200px"))
main_sort_order = widgets.ToggleButtons(options=["ASC", "DESC"], value="ASC", button_style="info")


# Bind events
table_selector.observe(preview_selected_tables, names='value')
run_button.on_click(run_query)
export_button.on_click(export_to_csv)
save_config_button.on_click(save_config)
load_config_button.on_click(load_config)
prebuilt_dropdown.observe(run_prebuilt, names='value')

# Sample connection setup (in practice, this should already be open)
conn = sqlite3.connect(":memory:")

# Sample table creation
conn.execute("CREATE TABLE example (id INTEGER PRIMARY KEY, name TEXT, age INTEGER)")
conn.executemany("INSERT INTO example (name, age) VALUES (?, ?)", [("Alice", 30), ("Bob", 25)])
conn.commit()

edit_output = widgets.Output()

def recreate_table_without_column(table_name, column_to_remove, max_rows=100):

    # Find primary key column
    cursor = conn.execute(f"PRAGMA table_info('{table_name}')")
    pk_cols = [row[1] for row in cursor if row[5] == 1]
    pk_column_ = pk_cols[0] if pk_cols else None

    if pk_column_ is None:
        # Fallback: use first column as order key
        temp_df = pd.read_sql_query(f"SELECT * FROM {table_name} LIMIT 1", conn)
        pk_column_ = temp_df.columns[0]

    # Load data ordered by primary key, limited by max_rows
    df = pd.read_sql_query(f"SELECT * FROM {table_name} ORDER BY {pk_column_} DESC LIMIT {max_rows}", conn)

    if column_to_remove not in df.columns:
        return f"❌ Column '{column_to_remove}' not found."

    new_df = df.drop(columns=[column_to_remove])

    # Create a string of column definitions (all TEXT for simplicity)
    cols_str = ', '.join([f"{col} TEXT" for col in new_df.columns])

    temp_table = f"{table_name}_new"

    try:
        conn.execute(f"DROP TABLE IF EXISTS {temp_table}")
        conn.execute(f"CREATE TABLE {temp_table} ({cols_str})")
        new_df.to_sql(temp_table, conn, if_exists='replace', index=False)
        conn.execute(f"DROP TABLE {table_name}")
        conn.execute(f"ALTER TABLE {temp_table} RENAME TO {table_name}")
        conn.commit()
        return f"✅ Column '{column_to_remove}' removed."
    except Exception as e:
        return f"❌ Error: {e}"


with edit_output:
    display(main_container)

# Persistent global list to hold rows widgets
if 'rows_widgets' not in globals():
    globals()['rows_widgets'] = []
rows_widgets = globals()['rows_widgets']


def adt_delete_tables(_):
    if not adt_table_selector.value:
        with adt_upload_output:
            print("⚠️ No tables selected to delete.")
        return

    if not confirm_delete_checkbox.value:
        with adt_upload_output:
            print("⚠️ Please confirm deletion by checking the box below.")
        return

    with adt_upload_output:
        clear_output()
        cursor = conn.cursor()
        for table in adt_table_selector.value:
            try:
                cursor.execute(f"DROP TABLE IF EXISTS {table}")
                print(f"🗑️ Deleted table '{table}'")
            except Exception as e:
                print(f"❌ Failed to delete table '{table}': {e}")
        conn.commit()
    adt_refresh_schema()
    adt_edit_output.clear_output()
    confirm_delete_checkbox.value = False

def editable_table_ui(table_name, max_rows=20, pk_column=None):
    print(f"🟢 Entering Edit Mode for table: {table_name}")
    main_ui_box.layout.display = 'none'
    edit_output.clear_output()
    edit_output.layout.display = 'block'

    global df
    try:
        df = pd.read_sql_query(f"SELECT * FROM {table_name} LIMIT {max_rows}", conn)
    except Exception as e:
        with edit_output:
            print(f"❌ Error loading table '{table_name}': {e}")
        return

    if df.empty:
        with edit_output:
            print("⚠️ Table is empty.")
        return

    if pk_column is None:
        cursor = conn.execute(f"PRAGMA table_info('{table_name}')")
        pk_cols = [row[1] for row in cursor if row[5] == 1]
        pk_column_ = pk_cols[0] if pk_cols else df.columns[0]
    else:
        pk_column_ = pk_column

    # === Filter & Sort Widgets ===
    sort_by_dropdown = widgets.Dropdown(
        options=df.columns.tolist(),
        description='Sort by:',
        layout=widgets.Layout(width='200px')
    )
    sort_order_toggle = widgets.ToggleButtons(
        options=['ASC', 'DESC'],
        value='ASC',
        button_style='info'
    )
    # Dropdown to choose the column to filter
    filter_column_dropdown = widgets.Dropdown(
        options=[None] + df.columns.tolist(),
        description='Filter by:',
        layout=widgets.Layout(width='200px')
    )

    # Dropdown to choose the value from selected column
    filter_value_dropdown = widgets.Dropdown(
        options=[],
        description='Value:',
        layout=widgets.Layout(width='200px')
    )

    # Global filtering/sorting widgets for main SQL preview
    main_filter_column = widgets.Dropdown(
        options=[],
        description="Filter by:",
        layout=widgets.Layout(width="200px")
    )
    main_filter_value = widgets.Dropdown(
        options=[],
        description="Value:",
        layout=widgets.Layout(width="200px")
    )
    main_sort_by = widgets.Dropdown(
        options=[],
        description="Sort by:",
        layout=widgets.Layout(width="200px")
    )
    main_sort_order = widgets.ToggleButtons(
        options=["ASC", "DESC"],
        value="ASC",
        button_style='info'
    )

    def update_main_filter_values(change):
        if not change["new"]:
            main_filter_value.options = []
            return

        selected_col = change["new"]
        if df is not None and selected_col in df.columns:
            unique_vals = df[selected_col].dropna().astype(str).unique()
            main_filter_value.options = [None] + sorted(unique_vals.tolist())

    # ---- Define your other widgets and handlers like refresh_button, save_btn, back_btn, etc. here ----

    main_container = widgets.VBox([
        widgets.HTML(f"<h4>📝 Editing Table: <code>{table_name}</code></h4>"),
        widgets.HBox([
            filter_column_dropdown,
            filter_value_dropdown,
            sort_by_dropdown,
            sort_order_toggle,
            # refresh_button,  # define this above
        ]),
  
        widgets.HTML("<b>🧹 Column Operations:</b>"),

    ])

    with edit_output:
        display(main_container)

        def on_main_apply_clicked(_):
            if df is None:
                return

            filtered_df = df.copy()

            col = main_filter_column.value
            val = main_filter_value.value
            if col and val:
                filtered_df = filtered_df[filtered_df[col].astype(str) == val]

            if main_sort_by.value:
                filtered_df = filtered_df.sort_values(
                    by=main_sort_by.value,
                    ascending=(main_sort_order.value == 'ASC')
                )

            output_box.clear_output()
            with output_box:
                display(filtered_df)


        def on_table_selected(change):
            table_name = change["new"][0] if change["new"] else None
            if not table_name:
                return

            try:
                global df
                df = pd.read_sql_query(f"SELECT * FROM {table_name} LIMIT 100", conn)
                if df.empty:
                    print("⚠️ Table is empty.")
                    return

                # Prepare column options
                cols = df.columns.tolist()
                main_filter_column.unobserve(update_main_filter_values, names='value')
                main_filter_column.options = [None] + cols
                main_filter_column.value = None
                main_filter_column.observe(update_main_filter_values, names='value')

                main_sort_by.options = cols
                main_sort_by.value = None
                main_sort_order.value = "ASC"

                orderby_dropdown.options = cols
                groupby_dropdown.options = [None] + cols

                main_filter_value.options = []

            except Exception as e:
                print(f"❌ Failed to load table '{table_name}': {e}")


        # Auto-update the second dropdown when column changes
        def update_filter_values(change):
            selected_col = change['new']
            if selected_col:
                unique_vals = df[selected_col].dropna().astype(str).unique()
                filter_value_dropdown.options = [None] + sorted(unique_vals.tolist())
            else:
                filter_value_dropdown.options = []

        filter_column_dropdown.observe(update_filter_values, names='value')

        refresh_button = widgets.Button(description='🔄 Apply Filter/Sort', button_style='primary')

        # Define UI containers
        scroll_container = widgets.VBox(layout=widgets.Layout(max_height='400px', overflow='auto'))

        # Prepare row widgets storage
        rows_widgets.clear()

        # === Helper Functions ===
        def apply_filter_sort():
            filtered_df = df.copy()

            # Use filter_column_dropdown and filter_value_dropdown instead of filter_text
            col = filter_column_dropdown.value
            val = filter_value_dropdown.value
            if col and val and val != '':  # Filter only if selection is valid
                filtered_df = filtered_df[filtered_df[col].astype(str) == val]

            if sort_by_dropdown.value:
                filtered_df = filtered_df.sort_values(
                    by=sort_by_dropdown.value,
                    ascending=(sort_order_toggle.value == 'ASC')
                )
            return filtered_df

        def rebuild_grid():
            grid = widgets.GridspecLayout(len(rows_widgets) + 1, len(df.columns) + 1, width='auto')
            for j, col in enumerate(df.columns):
                grid[0, j] = widgets.Label(col, layout=widgets.Layout(width='150px'))
            grid[0, len(df.columns)] = widgets.Label("🗑", layout=widgets.Layout(width='60px'))
            for i, (_, cells, del_cb, _, _) in enumerate(rows_widgets, start=1):
                for j, cell in enumerate(cells):
                    grid[i, j] = cell
                grid[i, len(df.columns)] = del_cb
            scroll_container.children = [grid]

        def add_row_at_position(position):
            new_cells = [widgets.Text(value='', layout=widgets.Layout(width='150px')) for _ in df.columns]
            del_cb = widgets.Checkbox(value=False, layout=widgets.Layout(width='60px'))
            new_row = (None, new_cells, del_cb, True, None)
            if position == 'top':
                rows_widgets.insert(0, new_row)
            else:
                rows_widgets.append(new_row)
            rebuild_grid()

        def on_save(_):
            insert_count = 0
            update_count = 0
            delete_count = 0
            cursor = conn.cursor()
            cursor.execute(f"PRAGMA table_info('{table_name}')")
            pk_info = [row for row in cursor if row[1] == pk_column_]
            pk_autoinc = pk_info and pk_info[0][2].upper() == 'INTEGER'

            try:
                for _, cells, del_cb, is_new, pk_val in rows_widgets:
                    if del_cb.value:
                        if not is_new and pk_val is not None:
                            cursor.execute(f"DELETE FROM {table_name} WHERE {pk_column_} = ?", (pk_val,))
                            delete_count += 1
                        continue

                    row_data = {
                        col: (cell.value.strip() if cell.value.strip() else None)
                        for col, cell in zip(df.columns, cells)
                    }

                    if is_new:
                        insert_cols = [col for col in row_data if not (pk_autoinc and col == pk_column_)]
                        if insert_cols:
                            cols = ', '.join(insert_cols)
                            placeholders = ', '.join(['?' for _ in insert_cols])
                            values = [row_data[col] for col in insert_cols]
                            cursor.execute(f"INSERT INTO {table_name} ({cols}) VALUES ({placeholders})", values)
                            insert_count += 1
                    else:
                        set_clause = ', '.join([f"{col} = ?" for col in row_data])
                        values = list(row_data.values()) + [pk_val]
                        cursor.execute(
                            f"UPDATE {table_name} SET {set_clause} WHERE {pk_column_} = ?", values)
                        update_count += 1

                conn.commit()
                print(f"💾 Saved: {insert_count} inserted, {update_count} updated, {delete_count} deleted.")
            except Exception as e:
                print(f"❌ Error saving changes: {e}")

                editable_table_ui(table_name)  # Refresh UI after save

        def on_go_back(_):
             # Clear edit output and hide edit UI
            edit_output.clear_output()
            edit_output.layout.display = 'none'

            # Show main UI again
            main_ui_box.layout.display = 'block'

            # Restore visibility of other main UI widgets as needed
            preview_box.layout.display = 'block'
            filter_box.layout.display = 'block'
            orderby_dropdown.layout.display = 'block'
            groupby_dropdown.layout.display = 'block'
            run_button.layout.display = 'block'
            export_button.layout.display = 'block'
            save_config_button.layout.display = 'block'
            load_config_button.layout.display = 'block'
            prebuilt_dropdown.layout.display = 'block'
            output_box.layout.display = 'block'
            table_selector.layout.display = 'block'
            upload_type.layout.display = 'block'
            upload_button.layout.display = 'block'
            upload_output.layout.display = 'block'
            # Reset the toggle so UI state stays consistent
            toggle_edit_mode.value = False
            # Attach handlers here
            save_btn.on_click(on_save)
            back_btn.on_click(on_go_back)

        def on_refresh_click(_):
            filtered = df.copy()

            selected_col = filter_column_dropdown.value
            selected_val = filter_value_dropdown.value

            # ✅ This is a comparison
            if selected_col and selected_val:
                filtered = filtered[filtered[selected_col].astype(str) == selected_val]

            # ✅ Sorting also correctly uses comparison
            if sort_by_dropdown.value:
                filtered = filtered.sort_values(
                    by=sort_by_dropdown.value,
                    ascending=(sort_order_toggle.value == 'ASC'))

            # Rebuild the UI rows
            rows_widgets.clear()
            for _, row in filtered.iterrows():
                cells = [widgets.Text(value='' if pd.isna(row[col]) else str(row[col]),
                                    layout=widgets.Layout(width='150px')) for col in df.columns]
                del_cb = widgets.Checkbox(value=False, layout=widgets.Layout(width='60px'))
                rows_widgets.append((None, cells, del_cb, False, row[pk_column_]))

            rebuild_grid()

        refresh_button.on_click(on_refresh_click)


        # === Filtered Data ===
        df = apply_filter_sort()

        # Populate rows_widgets
        for _, row in df.iterrows():
            cells = [widgets.Text(value='' if pd.isna(row[col]) else str(row[col]),
                                  layout=widgets.Layout(width='150px')) for col in df.columns]
            del_cb = widgets.Checkbox(value=False, layout=widgets.Layout(width='60px'))
            rows_widgets.append((None, cells, del_cb, False, row[pk_column_]))

        rebuild_grid()

        # === Column Operations ===
        col_delete_dropdown = widgets.Dropdown(
            options=[col for col in df.columns if col != pk_column_],
            description="Delete:", layout=widgets.Layout(width='250px'))
        drop_btn = widgets.Button(description="❌", button_style='danger', tooltip="Delete column")

        col_add_name = widgets.Text(description="Add:", placeholder="Column name")
        col_add_btn = widgets.Button(description="➕", button_style='info', tooltip="Add new column")

        def on_drop_column_clicked(_):
            msg = recreate_table_without_column(table_name, col_delete_dropdown.value)
            status_label.value = msg
            status_label.layout.display = 'block'
            editable_table_ui(table_name)

        drop_btn.on_click(on_drop_column_clicked)


        def on_add_column_clicked(_):
            col_name = col_add_name.value.strip()
            if not col_name:
                return
            try:
                conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {col_name} TEXT")
                conn.commit()
                status_label.value = f"➕ Column '{col_name}' added successfully."
                status_label.layout.display = 'block'
            except Exception as e:
                status_label.value = f"❌ Error adding column: {e}"
                status_label.layout.display = 'block'
            editable_table_ui(table_name)

        col_add_btn.on_click(on_add_column_clicked)

        # === Buttons ===
        save_btn = widgets.Button(description='💾 Save', button_style='success')
        back_btn = widgets.Button(description='↩️ Back', button_style='warning')
        insert_top_btn = widgets.Button(description='Insert Top', button_style='info')
        insert_bottom_btn = widgets.Button(description='Insert Bottom', button_style='info')

        save_btn.on_click(on_save)
        back_btn.on_click(on_go_back)
        insert_top_btn.on_click(lambda _: add_row_at_position('top'))
        insert_bottom_btn.on_click(lambda _: add_row_at_position('bottom'))
        refresh_button.on_click(on_refresh_click)

        buttons_top = widgets.HBox([save_btn, back_btn])
        buttons_bottom = widgets.HBox([insert_top_btn, insert_bottom_btn])
        col_ops = widgets.HBox([
            col_delete_dropdown,  # dropdown to select column to delete
            drop_btn,             # button to drop the column
            col_add_name,         # text input to type new column name
            col_add_btn           # button to add the new column
        ])


        # === Show UI ===
        main_container.children = [
            widgets.HTML("<b>🔍 Filter & Sort Options:</b>"),
            widgets.HBox([filter_column_dropdown, filter_value_dropdown, sort_by_dropdown, sort_order_toggle, refresh_button]),
            widgets.HTML("<b>🧹 Column Operations:</b>"),
            col_ops,
            widgets.HTML("<hr>"),
            buttons_top,
            buttons_bottom,
            status_label,  # 👈 Add this
            scroll_container,
        ]

# --- Add/Delete Table Mode widgets and logic ---
add_delete_mode_toggle = widgets.ToggleButton(
    value=False,
    description='Add/Delete Table Mode',
    tooltip='Toggle Add/Delete Table Mode',
    button_style='warning',
    layout=widgets.Layout(width='180px'))

adt_upload_type = widgets.ToggleButtons(
    options=['📁 Upload .db file', '📊 Upload Excel/CSV'],
    description='Data Source:',
    button_style='info')

adt_upload_button = widgets.Button(description='Upload (Add/Delete)', button_style='primary')
adt_upload_output = widgets.Output()

adt_table_selector = widgets.SelectMultiple(
    options=[],
    description='Tables (Add/Delete):',
    layout=widgets.Layout(width='50%'))

adt_delete_table_btn = widgets.Button(description='Delete Selected Table(s)', button_style='danger')
adt_edit_output = widgets.Output()

def adt_refresh_schema():
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [t[0] for t in cur.fetchall()]
    adt_table_selector.options = tables

def adt_delete_tables(_):
    if not adt_table_selector.value:
        with adt_upload_output:
            print("⚠️ No tables selected to delete.")
        return
    with adt_upload_output:
        clear_output()
        cursor = conn.cursor()
        for table in adt_table_selector.value:
            try:
                cursor.execute(f"DROP TABLE IF EXISTS {table}")
                print(f"🗑️ Deleted table '{table}'")
            except Exception as e:
                print(f"❌ Failed to delete table '{table}': {e}")
        conn.commit()
    adt_refresh_schema()
    adt_edit_output.clear_output()

def adt_upload_handler(_):
    global conn, DB_PATH
    adt_upload_output.clear_output()
    with adt_upload_output:
        if adt_upload_type.value == '📁 Upload .db file':
            DB_PATH = upload_existing_db()
            if DB_PATH is None:
                print("❌ No DB uploaded.")
                return
            try:
                conn = sqlite3.connect(DB_PATH)
                print(f"✅ Connected to DB {DB_PATH}")
            except Exception as e:
                print(f"❌ Failed to open DB: {e}")
                return
        elif adt_upload_type.value == '📊 Upload Excel/CSV':
            temp_db_path = "/content/auto_generated_add_delete.db"
            DB_PATH = create_db_from_excel(temp_db_path)
            if DB_PATH is None:
                print("❌ Failed to create DB from Excel/CSV.")
                return
            conn = sqlite3.connect(DB_PATH)
            print(f"✅ Created new DB from Excel/CSV: {DB_PATH}")
        adt_refresh_schema()

adt_upload_button.on_click(adt_upload_handler)
adt_delete_table_btn.on_click(adt_delete_tables)

def adt_edit_table(change):
    adt_edit_output.clear_output()
    with adt_edit_output:
        if not conn:
            print("⚠️ No DB connected.")
            return
        if not adt_table_selector.value:
            print("⚠️ Select a table to edit.")
            return
        # Show editable UI for the first selected table only
        editable_table_ui(adt_table_selector.value[0])

    editable_table_ui(table_selector.value[0])

adt_table_selector.observe(adt_edit_table, names='value')

def on_add_delete_mode_toggle(change):
    if change['new']:
        # Enter Add/Delete Table Mode
        # Hide main UI widgets
        preview_box.layout.display = 'none'
        filter_box.layout.display = 'none'
        orderby_dropdown.layout.display = 'none'
        groupby_dropdown.layout.display = 'none'
        run_button.layout.display = 'none'
        export_button.layout.display = 'none'
        save_config_button.layout.display = 'none'
        load_config_button.layout.display = 'none'
        prebuilt_dropdown.layout.display = 'none'
        output_box.layout.display = 'none'
        table_selector.layout.display = 'none'
        upload_type.layout.display = 'none'
        upload_button.layout.display = 'none'
        upload_output.layout.display = 'none'

        # Show Add/Delete Mode widgets
        add_delete_mode_toggle.layout.display = 'block'
        adt_upload_type.layout.display = 'flex'
        adt_upload_button.layout.display = 'inline-block'
        adt_upload_output.layout.display = 'block'
        adt_table_selector.layout.display = 'block'
        adt_delete_table_btn.layout.display = 'inline-block'
        adt_edit_output.layout.display = 'block'

        edit_output.clear_output()
    else:
        # Exit Add/Delete Table Mode, show main UI widgets
        preview_box.layout.display = None
        filter_box.layout.display = None
        orderby_dropdown.layout.display = None
        groupby_dropdown.layout.display = None
        run_button.layout.display = None
        export_button.layout.display = None
        save_config_button.layout.display = None
        load_config_button.layout.display = None
        prebuilt_dropdown.layout.display = None
        output_box.layout.display = None
        table_selector.layout.display = None
        upload_type.layout.display = None
        upload_button.layout.display = None
        upload_output.layout.display = None

        # Hide Add/Delete Mode widgets
        adt_upload_type.layout.display = 'none'
        adt_upload_button.layout.display = 'none'
        adt_upload_output.layout.display = 'none'
        adt_table_selector.layout.display = 'none'
        adt_delete_table_btn.layout.display = 'none'
        adt_edit_output.layout.display = 'none'

        edit_output.clear_output()

add_delete_mode_toggle.observe(on_add_delete_mode_toggle, names='value')

# --- Initialize Add/Delete Mode widgets as hidden ---
adt_upload_type.layout.display = 'none'
adt_upload_button.layout.display = 'none'
adt_upload_output.layout.display = 'none'
adt_table_selector.layout.display = 'none'
adt_delete_table_btn.layout.display = 'none'
adt_edit_output.layout.display = 'none'

# ===== Toggle button for editing selected tables (your original "Edit Mode") =====
toggle_edit_mode = widgets.ToggleButton(
    value=False,
    description='Edit Selected Table',
    tooltip='Toggle Edit Selected Table Mode',
    button_style='warning',
    layout=widgets.Layout(width='150px')
)

def on_toggle_edit(change):
    if change['new']:
        print("🟡 Edit Mode activated")
        print("🔎 Available tables:", table_selector.options)
        print("📌 Selected table:", table_selector.value)

        # Enter edit mode (main UI)
        preview_box.layout.display = 'block'
        filter_box.layout.display = 'none'
        orderby_dropdown.layout.display = 'none'
        groupby_dropdown.layout.display = 'none'
        run_button.layout.display = 'none'
        export_button.layout.display = 'none'
        save_config_button.layout.display = 'none'
        load_config_button.layout.display = 'none'
        prebuilt_dropdown.layout.display = 'none'
        output_box.layout.display = 'none'
        table_selector.layout.display = 'none'
        upload_type.layout.display = 'none'
        upload_button.layout.display = 'none'
        upload_output.layout.display = 'none'

        edit_output.layout.display = 'block'

        selected = table_selector.value
        if selected and isinstance(selected, tuple) and len(selected) > 0:
            editable_table_ui(selected[0])  # ✅ Safe usage
        else:
            with edit_output:
                print("⚠️ Select a table to edit.")

    else:
        # Exit edit mode, show main UI widgets
        preview_box.layout.display = 'block'  # or 'flex'
        filter_box.layout.display = None
        orderby_dropdown.layout.display = None
        groupby_dropdown.layout.display = None
        run_button.layout.display = None
        export_button.layout.display = None
        save_config_button.layout.display = None
        load_config_button.layout.display = None
        prebuilt_dropdown.layout.display = None
        output_box.layout.display = None
        table_selector.layout.display = None
        upload_type.layout.display = None
        upload_button.layout.display = None
        upload_output.layout.display = None

        edit_output.layout.display = 'none'
        edit_output.clear_output()

toggle_edit_mode.observe(on_toggle_edit, names='value')

# --- Initialize edit output hidden ---
edit_output.layout.display = 'none'
# Success / status message label (initially hidden)
status_label = widgets.Label()
status_label.layout.display = 'none'

# ===== Display UI =====
# === Rebuild main UI box ===
main_ui_box.children = [
    widgets.HTML("<h3>🔄 Choose Data Source and Upload</h3>"),
    upload_type,
    upload_button,
    upload_output,
    widgets.HTML("<h3>\U0001F4CA Interactive Sales DB Explorer</h3>"),
    widgets.HBox([toggle_edit_mode, add_delete_mode_toggle]),
    widgets.HBox([table_selector]),
    preview_box,
    widgets.HTML("<b>\U0001F50D Filters:</b>"),
    filter_box,
    widgets.HBox([orderby_dropdown, groupby_dropdown]),
    widgets.HBox([run_button, export_button, save_config_button, load_config_button]),
    widgets.HTML("<b>\U0001F4A1 Pre-Built Reports:</b>"),
    widgets.HTML("<b>🔍 Filter & Sort (Live Preview):</b>"),
    widgets.HBox([main_filter_column, main_filter_value]),
    prebuilt_dropdown,
    output_box
]

# Now display the 3 major layout containers
display(main_ui_box)       # Main UI
display(edit_output)       # Edit Mode UI
display(adt_upload_type)   # Add/Delete UI
display(adt_upload_button)
display(adt_upload_output)
display(adt_table_selector)
display(adt_delete_table_btn)
display(adt_edit_output)

adt_upload_type.layout.display = 'none'
adt_upload_button.layout.display = 'none'
adt_upload_output.layout.display = 'none'
adt_table_selector.layout.display = 'none'
adt_delete_table_btn.layout.display = 'none'
adt_edit_output.layout.display = 'none'

# Show main UI initially
main_ui_box.layout.display = 'block'
edit_output.layout.display = 'none'


# --- Initialize globals ---
conn = None
DB_PATH = None
tables = []
columns = []

# Preview tables if already available
if tables:
    preview_selected_tables({'new': table_selector.value})
