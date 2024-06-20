# Recipes Recommendation Service

This is a repository for the Recipe Recommendation Service using Python and FastAPI. This service allows users to receive recipe recommendations based on the ingredients they have.

## Features

-
-

## Directory Structure

```bash
- app               
-- model.py         # Module to load model
-- server.py        # Main module to run the FastAPI service
- Dockerfile        # Docker configuration to build the image
- requirements.txt  # List of required dependencies
```

## Installation

- Make sure you have Python 3.x installed on your local machine.
- Clone this repository to your local machine.

```bash
git clone https://github.com/Reseepe/recommendation-system.git
```

### Activate the virtual environment: 

```bash
python -m venv venv
```

### Install the project dependencies:

```bash
pip install -r requirements.txt
```

### Start

```bash
uvicorn app.server:app --port 8080
```


## API Reference

#### Get Recommendation Recipes

```http
    POST /recommendaton
```

| Parameter     | Type      | Description                           |
| :--------     | :---      | :==========                           |
| `user_input`  | `string`  | **Required**. Your ingredient list    |

Example Request

```json
{
    "user_input": "water, chicken"
}
```

Example Response

```json
{
    "recommended_recipes": [
        "stuffed chicken in a blue cheese and pecan sauce",
        "high roasted chicken and potatoes",
        "chinese roasted chicken",
    ]
}
```

Bangkit Team C241-PS428

Bangkit Academy 2024 batch 1
