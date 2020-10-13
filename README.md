# DTMF Generator
Python application to generate DTMF tones to phone numbers.

## Install

To facilitate the install process you can use [Pipenv](https://github.com/pypa/pipenv) and execute the following command to create the virtual environment with Python dependencies to execute the application:

```
pipenv install
```

## Run

You can run the application using:

```
pipenv run python dtmf-generator.py -p PHONE_NUMBER -f SAMPLE_FREQ -t TONE_DURATION -s SILENCE_DURATION -a AMPLITUDE -o FILENAME_OUTPUT -d
```

Where `PHONE_NUMBER` represents the phone number that you want convert to DMTF tones, `SAMPLE_FREQ` is the sampling frequency to generate tones, `TONE_DURATION` defines the duration in seconds for each tone, `SILENCE_DURATION` defines the duration in seconds for each silence between tones, `AMPLITUDE` defines the amplitude of the sine wave signal, `FILENAME_OUTPUT` is the filename to WAV file output, and `-d` is an **optional** flag to activate the debug mode which enable a graph with the frequency plot of the tones perceived in each time. For example you can use the following:

```
pipenv run python dtmf-generator.py -p 12345 -f 20000 -t 0.08 -s 0.08 -o test.wav -d
```

You can verify if the tones were correctly generated using the [DialABC site](http://dialabc.com/sound/detect/index.html), where you should upload the output of DTMF Generator. You should observe the same phone number input in the site output.
