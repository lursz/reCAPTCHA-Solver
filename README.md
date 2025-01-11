# CAPTCHA Solving
## About
CAPTCHA is an acronym for Completely Automated Turing Test to Tell Computers and Humans Apart. As the name suggests, it is a system designed solely to distinguish between humans and machines. This means that CAPTCHA challenges ought to be easily solvable for humans, but impossible to solve for any automated systems.

This repository explores the feasibility of breaking image-based reCAPTCHA challenges using machine learning techniques. The aim is to proove that reCAPTCHA is no longer safe, and can be broken with relatively mild effort, even without using transformer based models.

[gifs]

## Essence
Currently, breaking reCAPTCHA systems primarily involves solving two core challenges. Naturally, there are additional ones, but all can be addressed using models trained for these core tasks:
- Multi Image CAPTCHAs
- Single Image CAPTCHAs
[pic1][pic2]


# How to use
## Installation
In order to run the program, you will need to install `Python3.12` or greater and `pip`. Then install the required dependencies by typing:
```bash
pip install -r requirements.txt
```

## Running the program
Begin with editing `.env` file. Fill all the needed data - browser type and paths to files containing model weights.
Then activate your docker container with:
```bash
cd docker
sudo docker compose up
```
and run the program by typing:
```bash
python main.py
```

# How it all works?


