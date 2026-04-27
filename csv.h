/**
 * @file csv.h
 * @brief Header-only CSV parsing library
 *
 * A lightweight, self-contained CSV parser written in C. This library provides
 * functions to parse CSV strings/files into an in-memory row-oriented
 * structure.
 *
 * Features:
 * - RFC 4180 compliant CSV parsing
 * - Quoted fields with escaped quotes ("")
 * - Optional whitespace trimming
 * - CRLF (\r\n) and LF (\n) line endings
 * - Memory-efficient dynamic growth
 *
 * Usage:
 * @code
 * #define CSV_IMPLEMENTATION
 * #include "csv.h"
 *
 * csv_t* csv = csv_load("data.csv");
 * for (size_t i = 0; i < csv->size; i++) {
 *     for (size_t j = 0; j < csv->rows[i].size; j++) {
 *         printf("%s ", csv->rows[i].fields[j]);
 *     }
 * }
 * free_csv_data(csv);
 * free(csv);
 * @endcode
 *
 * @license MIT
 */

#ifndef CSV_H
#define CSV_H

#include <ctype.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ============================================================================
 * DATA STRUCTURES
 * ============================================================================ */

/**
 * @brief CSV parser context
 *
 * Maintains the state during CSV parsing. The cursor points to the current
 * position in the input string being parsed.
 *
 * @note Error codes:
 *       0 = no error
 *       1 = error occurred (see error_msg for details)
 */
typedef struct csv_ctx {
    const char* csv;       /**< Original CSV string (for reference) */
    const char* cursor;   /**< Current parsing position */
    int error_code;       /**< Error code (0 = success, 1 = error) */
    char* error_msg;      /**< Detailed error message (caller must free) */
    bool trim_whitespaces;/**< Whether to trim leading/trailing whitespace */
} csv_ctx;

/**
 * @brief A single row in the CSV
 *
 * Contains an array of field strings. Each field is owned by this structure
 * and must be freed when the row is freed.
 */
typedef struct csv_row {
    char** fields;   /**< Array of field strings */
    size_t size;     /**< Number of fields in this row */
    size_t capacity; /**< Allocated capacity for fields array */
} csv_row;

/**
 * @brief Complete CSV data structure
 *
 * Row-oriented representation of a CSV file. The first row is typically
 * the header, but this library treats all rows uniformly.
 *
 * @note The rows array is dynamically allocated and grows as needed
 */
typedef struct csv_t {
    csv_row* rows;   /**< Array of rows */
    size_t size;     /**< Number of rows */
    size_t capacity; /**< Allocated capacity for rows array */
} csv_t;

/* ============================================================================
 * CONTEXT MANAGEMENT
 * ============================================================================ */

/**
 * @brief Initialize a CSV parsing context
 *
 * Sets up the context for parsing. After calling this, the context
 * is ready to use with parse_csv().
 *
 * @param[out] ctx      Pointer to context to initialize
 * @param[in]  csv_data String containing CSV data to parse
 *
 * @note The csv_data string must remain valid until parsing is complete
 */
static void init_csv_context(csv_ctx* ctx, const char* csv_data);

/* ============================================================================
 * MEMORY MANAGEMENT
 * ============================================================================ */

/**
 * @brief Free a single CSV row and all its fields
 *
 * Releases all memory associated with a CSV row, including all field
 * strings and the fields array itself.
 *
 * @param[in,out] row Pointer to the row to free
 *
 * @note Safe to call with NULL pointer (no-op)
 */
static void free_csv_row(csv_row* row);

/**
 * @brief Free an entire CSV structure
 *
 * Releases all memory associated with a CSV structure, including all rows
 * and all field strings within those rows.
 *
 * @param[in,out] csv_data Pointer to the CSV structure to free
 *
 * @note Safe to call with NULL pointer (no-op)
 * @note Only frees the internal arrays; caller must free csv_t itself
 *
 * @par Example:
 * @code
 * csv_t* csv = csv_load("data.csv");
 * free_csv_data(csv);   // Frees all rows and fields
 * free(csv);            // Frees the csv_t structure
 * @endcode
 */
static void free_csv_data(csv_t* csv_data);

/* ============================================================================
 * PARSING HELPERS
 * ============================================================================ */

/**
 * @brief Skip whitespace characters
 *
 * Advances the cursor past any leading whitespace characters (space, tab, etc.)
 * at the current position.
 *
 * @param[in,out] ctx Pointer to parsing context
 *
 * @note Uses isspace() from ctype.h, which handles locale-specific whitespace
 */
static void skip_whitespaces(csv_ctx* ctx);

/**
 * @brief Append a character to a dynamically growing string
 *
 * Helper function that appends a single character to a string buffer,
 * automatically growing the buffer as needed (doubling strategy).
 *
 * @param[in,out] str     Pointer to the current string (may be NULL for new
 * string)
 * @param[in,out] length Pointer to current string length (updated in place)
 * @param[in,out] capacity Pointer to allocated capacity (updated in place)
 * @param[in]     ch      Character to append
 *
 * @return Updated string pointer (may change after realloc)
 * @return NULL if realloc fails (original string is freed)
 *
 * @note When *capacity is 0, a new buffer of 16 bytes is allocated
 * @note The buffer is grown by doubling (*capacity << 1) when full
 */
static char* string_append(char* str, size_t* length, size_t* capacity,
                           char ch);

/**
 * @brief Append a new row to the CSV structure
 *
 * Dynamically grows the rows array if needed, then initializes a new
 * empty row at the end.
 *
 * @param[in,out] csv Pointer to CSV structure to append to
 *
 * @return Pointer to the new row, or NULL if realloc fails
 *
 * @note The returned row's fields array is initially empty
 * @note Growth strategy: start at 3, then double
 */
static csv_row* rows_append(csv_t* csv);

/* ============================================================================
 * CORE PARSING FUNCTIONS
 * ============================================================================ */

/* Forward declaration for mutual recursion with parse_csv_row */
static char* parse_csv_field(csv_ctx* ctx);

/**
 * @brief Parse a single row from the CSV
 *
 * Reads fields from the current cursor position until a newline is reached
 * or end of string. Each field is parsed using parse_csv_field().
 *
 * Line ending handling:
 * - LF (\n): advances cursor past the newline
 * - CRLF (\r\n): advances cursor past both characters
 *
 * @param[in,out] ctx Pointer to parsing context
 *
 * @return Pointer to newly allocated csv_row, or NULL on error
 *
 * @par CSV Row Format:
 * @code
 * field1,field2,"field with , comma",field4
 * @endcode
 *
 * @see parse_csv_field()
 * @see parse_csv()
 */
static csv_row* parse_csv_row(csv_ctx* ctx);

/**
 * @brief Parse a single CSV field
 *
 * Parses one field from the current cursor position. Handles:
 * - Quoted fields: "content", allowing commas inside
 * - Escaped quotes: "say ""hello""" -> say "hello"
 * - Unquoted fields: stop at comma or newline
 * - Leading/trailing whitespace trimming for unquoted fields
 *
 * State Machine:
 * @code
 *      +-------+
 *      | START |
 *      +---+---+
 *          |
 *          v
 *      +---------------+
 *      | READ CHAR     |<-+
 *      +---+---+-------+  |
 *          |   |       |
 *          |   | (next char)
 *          |   +-------+
 *          |
 *          v
 *      +-----+-------+-------+===============
 *      |"\\n"| "\\r" | MORE  |   QUOTE      |
 *      +-----+-------+-------+===============
 *      |END  | END   | BREAK | CHECK        |
 *      +-----+-------+-------+              |
 *                                   +-------+-------+
 *                                   |  in_quotes?   |
 *                                   +---+-------++--+
 *                                       |       |   |
 *                                       | YES   |NO |
 *                                       v       v   |
 *                                  +--------+  +----------+
 *                                  |ESCAPED |  | TOGGLE   |
 *                                  |QUOTE?  |  | in_quotes|
 *                                  +---+----+  +----+-----+
 *                                      |            |
 *                                      v            v
 *                                   +-------+  +-------+
 *                                   |SKIP & |  | BREAK |
 *                                   |APPEND"|  +-------+
 *                                   +-------+
 * @endcode
 *
 * @param[in,out] ctx Pointer to parsing context
 *
 * @return Newly allocated string containing the field, or "" for empty fields
 * @return NULL if error occurs (e.g., unclosed quote, memory failure)
 *
 * @par Quoted Field Examples:
 * @code
 * "simple"           -> simple
 * "with , comma"     -> with , comma
 * "say ""hi"""       -> say "hi"
 * @endcode
 *
 * @note After parsing, cursor points to the character after the field
 */
static char* parse_csv_field(csv_ctx* ctx);

/**
 * @brief Parse an entire CSV string
 *
 * Main entry point for parsing CSV data. Takes a string and returns
 * a fully populated csv_t structure.
 *
 * @param[in] input Null-terminated string containing CSV data
 *
 * @return Newly allocated csv_t structure, or NULL on error
 *
 * @par Input Format:
 * @code
 * header1,header2,header3
 * value1,value2,value3
 * "quoted, value",value4,value5
 * @endcode
 *
 * @note The input string is not modified; parsing is done read-only
 * @note Empty lines at the end of input are skipped
 * @note Whitespace trimming is enabled by default
 */
static csv_t* parse_csv(const char* input);

/**
 * @brief Load and parse a CSV file
 *
 * Opens a file, reads its entire contents into memory, and parses
 * the CSV data.
 *
 * @param[in] filepath Path to the CSV file
 *
 * @return Newly allocated csv_t structure, or NULL on error
 *
 * @note The file is closed after reading
 * @note File size is determined by fseek/ftell
 *
 * @par Example:
 * @code
 * csv_t* csv = csv_load("data.csv");
 * if (!csv) { perror("Failed"); return 1; }
 * // ... use csv ...
 * free_csv_data(csv);
 * free(csv);
 * @endcode
 */
static csv_t* csv_load(const char* filepath);

/* ============================================================================
 * IMPLEMENTATION
 * ============================================================================ */

#ifdef CSV_IMPLEMENTATION

static void init_csv_context(csv_ctx* ctx, const char* csv_data) {
    ctx->csv = csv_data;
    ctx->cursor = csv_data;
    ctx->error_code = 0;
    ctx->error_msg = NULL;
}

static void free_csv_row(csv_row* row) {
    if (!row)
        return;

    /* Free each individual field string */
    for (size_t i = 0; i < row->size; ++i)
        free(row->fields[i]);

    /* Free the fields array itself */
    free(row->fields);

    /* Reset row state to prevent double-free */
    row->fields = NULL;
    row->size = 0;
    row->capacity = 0;
}

static void free_csv_data(csv_t* csv_data) {
    if (!csv_data)
        return;

    /* Free each row and its contents */
    for (size_t i = 0; i < csv_data->size; ++i)
        free_csv_row(&csv_data->rows[i]);

    /* Free the rows array */
    free(csv_data->rows);

    /* Reset state */
    csv_data->rows = NULL;
    csv_data->size = 0;
    csv_data->capacity = 0;
}

static void skip_whitespaces(csv_ctx* ctx) {
    while (ctx->cursor && isspace((int)*ctx->cursor))
        ++ctx->cursor;
}

static char* string_append(char* str, size_t* length, size_t* capacity,
                           char ch) {
    /* Grow buffer if needed (leave 1 byte for null terminator) */
    if (*capacity == 0 || *length >= *capacity - 1) {
        /* Initial capacity of 16, then double */
        size_t new_capacity = (*capacity == 0) ? 16 : (*capacity) << 1;
        char* new_string = (char*)realloc(str, new_capacity);
        if (!new_string) {
            return NULL;
        }
        str = new_string;
        *capacity = new_capacity;
    }

    /* Append character and update length */
    str[(*length)++] = ch;
    return str;
}

static csv_row* rows_append(csv_t* csv) {
    /* Grow rows array if needed */
    if (csv->size >= csv->capacity) {
        size_t new_capacity = (csv->capacity == 0) ? 3 : (csv->capacity) << 1;
        csv_row* new_rows =
            (csv_row*)realloc(csv->rows, sizeof(csv_row) * new_capacity);
        if (new_rows == NULL) {
            return NULL;
        }
        csv->rows = new_rows;
        csv->capacity = new_capacity;
    }

    /* Initialize new row at end */
    csv_row* new_row = &csv->rows[csv->size++];
    new_row->fields = NULL;
    new_row->size = 0;
    new_row->capacity = 0;
    return new_row;
}

static csv_row* parse_csv_row(csv_ctx* ctx) {
    /* Allocate new row */
    csv_row* row = (csv_row*)malloc(sizeof(csv_row));
    if (!row) {
        return NULL;
    }

    row->fields = NULL;
    row->size = 0;
    row->capacity = 0;

    /* Parse fields until newline or end of input */
    while (*ctx->cursor && *ctx->cursor != '\n' && *ctx->cursor != '\r') {
        /* Skip leading whitespace before each field */
        skip_whitespaces(ctx);

        /* Check for empty field or end of line */
        if (*ctx->cursor == '\n' || *ctx->cursor == '\r' ||
            *ctx->cursor == '\0') {
            break;
        }

        /* Parse the field */
        char* field = parse_csv_field(ctx);
        if (!field && ctx->error_code) {
            free_csv_row(row);
            return NULL;
        }

        /* Grow fields array and append new field */
        char** new_fields =
            (char**)realloc(row->fields, sizeof(char*) * (row->size + 1));
        if (!new_fields) {
            free(field);
            free_csv_row(row);
            return NULL;
        }
        row->fields = new_fields;
        row->fields[row->size++] = field;
    }

    /* Handle line endings: skip CR, LF, or CRLF */
    if (*ctx->cursor == '\r')
        ++ctx->cursor;
    if (*ctx->cursor == '\n')
        ++ctx->cursor;

    return row;
}

static char* parse_csv_field(csv_ctx* ctx) {
    char* field = NULL;
    size_t length = 0, capacity = 0;
    bool in_quotes = false;        /* Whether we're inside quotes */
    bool is_start_of_field = true; /* For leading whitespace trimming */
    bool is_quoted_field = false;  /* Whether field was quoted */

    while (*ctx->cursor) {
        char ch = *ctx->cursor;

        /* Handle newline: end field (only when not in quotes) */
        if (ch == '\n' || ch == '\r') {
            break;
        }

        /* Handle quote character */
        if (ch == '"') {
            /* Check for escaped quote ("") */
            if (in_quotes && ctx->cursor[1] == '"') {
                /* Skip second quote, append single quote to field */
                ++ctx->cursor;
                field = string_append(field, &length, &capacity, '"');
                if (!field) {
                    return NULL;
                }
            } else {
                /* Toggle quote state */
                in_quotes = !in_quotes;
                if (in_quotes)
                    is_quoted_field = true;
            }
        }
        /* Handle comma: end of field (only when not in quotes) */
        else if (ch == ',' && !in_quotes) {
            ++ctx->cursor;
            break;
        }
        /* Handle regular characters */
        else {
            /* Skip leading whitespace for unquoted fields */
            if (is_start_of_field && isspace(ch) && ctx->trim_whitespaces &&
                !is_quoted_field) {
                /* Skip this whitespace character */
            } else {
                /* Append character to field */
                field = string_append(field, &length, &capacity, ch);
                if (!field) {
                    return NULL;
                }

                /* Only non-whitespace ends the "leading whitespace" state */
                if (!isspace(ch)) {
                    is_start_of_field = false;
                }
            }
        }

        ++ctx->cursor;
    }

    /* Trim trailing whitespace for unquoted fields */
    if (field && !is_quoted_field && ctx->trim_whitespaces && length > 0) {
        while (length > 0 && isspace(field[length - 1])) {
            --length;
        }
        field[length] = '\0';
    }

    /* Error: unclosed quote */
    if (in_quotes) {
        free(field);
        return NULL;
    }

    /* Return empty string for blank fields */
    if (!field) {
        field = (char*)malloc(1);
        if (field)
            field[0] = '\0';
    }

    return field;
}

static csv_t* parse_csv(const char* input) {
    csv_ctx ctx;
    init_csv_context(&ctx, input);
    ctx.trim_whitespaces = true;

    /* Allocate CSV structure */
    csv_t* csv = (csv_t*)malloc(sizeof(csv_t));
    if (!csv) {
        return NULL;
    }
    csv->rows = NULL;
    csv->size = 0;
    csv->capacity = 0;

    /* Parse rows until end of input */
    while (*ctx.cursor) {
        /* Skip empty lines */
        while (*ctx.cursor == '\n' || *ctx.cursor == '\r') {
            ++ctx.cursor;
        }

        if (*ctx.cursor == '\0') {
            break;
        }

        /* Parse a single row */
        csv_row* row = parse_csv_row(&ctx);
        if (!row && ctx.error_code) {
            free_csv_data(csv);
            free(csv);
            return NULL;
        }

        if (row) {
            csv_row* appended = rows_append(csv);
            if (!appended) {
                free_csv_row(row);
                free_csv_data(csv);
                free(csv);
                return NULL;
            }
            *appended = *row;
            free(row);
        }
    }

    return csv;
}

static csv_t* csv_load(const char* filepath) {
    FILE* file = fopen(filepath, "r");
    if (!file) {
        return NULL;
    }

    /* Determine file size */
    fseek(file, 0, SEEK_END);
    long file_size = ftell(file);
    fseek(file, 0, SEEK_SET);

    /* Allocate buffer */
    char* buffer = (char*)malloc(file_size + 1);
    if (!buffer) {
        fclose(file);
        return NULL;
    }

    /* Read file contents */
    fread(buffer, 1, file_size, file);
    buffer[file_size] = '\0';
    fclose(file);

    /* Parse CSV data */
    csv_t* csv = parse_csv(buffer);
    free(buffer);

    return csv;
}

#endif /* CSV_IMPLEMENTATION */

#endif /* CSV_H */
