# Zeus chatbot
*Project for Methods in AI Research*

Zeus is a goal-based chatbot that helps users find restaurants.

By Florian van der Steen, Eleni Veroni, Colino Sprockel and Sebastiaan Jans

## Requirements
- [keras 2.6.0](https://keras.io/)
- [numpy 1.19.5](https://numpy.org/install/)
- [pyttsx3 2.90](https://pypi.org/project/pyttsx3/)
- [nltk 3.6.3](https://www.nltk.org/install.html)
- [pandas 1.2.0](https://pandas.pydata.org/getting_started.html)

## Usage
Run the ```main.py``` file using ```python3```. Zeus has  main command line argument options, namely ```--baseline``` and ```--t2s```. Using ```--baseline``` the algorithm that classifies the user input is changed to a simplistic model based on direct matching. Using ```--t2s```, text will also be uttered through the audio output. A third option, ```--debug```, enables debug print statements.

## Example
```
python3 main.py --baseline

Welcome to Zeus bot, let me help you suggest a restaurant, do you have any preferences?
```
