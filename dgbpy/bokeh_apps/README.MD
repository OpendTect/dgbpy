# Bokeh-Apps

## Running from a command prompt
The Bokeh apps included in dgbpy were developed to be launched from the OpendTect
main application. However, especially during development, it might be preferred to
run them from a command prompt. The simplest scenario is to invoke the commands
in a terminal/command prompt opened from inside OpendTect as that ensures the
active Python environment will be used and the python folder of the OpendTect
software installation, where odpy and dgbpy live, will be included in the PYTHONPATH.
Note that OpendTect can be closed after the terminal/command prompt starts.
If you want to start the application completely outside of OpendTect it's up to
you to ensure the active Python environment has the dependencies required by the
app, that the PYTHONPATH is correctly set so everything can be found and the
location of the OpendTect data and the survey to use are provided.

Some of the apps depend on network communication with OpendTect to get input
parameters but it is possible to run them by passing a parameter string on the
command line. Typically a command line will look like:

````
bokeh serve --show "path to app folder" --args "parameter string"
````
where the "parameter string" is a JSON format string. The specific content of the
parameter string is different for each app and is described below. It is critical
that the JSON parameter string is "escaped" so it is not broken up by the command
shell you are using. This is different for each operating system.

### Linux/MacOS - Bash
````
'{"Examples File": "/ODData/F3_Demo_2020/NLAs/test_img2imgreg_data.h5"}'
or
"{\"Examples File\": \"/ODData/F3_Demo_2020/NLAs/test_img2imgreg_data.h5\"}"
````

### Windows - Command Prompt
````
"{\"Examples File\": \"C:\\ODData\\F3_Demo_2020\\NLAs\\test_img2imgreg_data.h5\"}"
or
"{\"Examples File\": \"C:/ODData/F3_Demo_2020/NLAs/test_img2imgreg_data.h5\"}"
````

### Windows - Powershell
````
'{\"Examples File\": \"C:\\ODData\\F3_Demo_2020\\NLAs\\test_img2imgreg_data.h5\"}'
or
'{\"Examples File\": \"C:/ODData/F3_Demo_2020/NLAs/test_img2imgreg_data.h5\"}'
````

## Crossplot App
The crossplot app generates multi-well crossplots of wireline log data. It is a
completely stand-a-lone app in its present form and so doesn't require a parameter
string. To start this app enter something like:

````
bokeh serve --show  /opt/OpendTect/6.6.0/bin/python/dgbpy/bokeh_apps/crossplot
````
where the location of the crossplot folder will depend on your installation. This
should open a new web browser tab in your default system web browser displaying
the app. Note as with all Bokeh applications you can copy the url into another
new tab to open a new separate version of the application.

## Example Viewer App
The example viewer app is used to browse the training data for Img2Img machine
learning workflows. The app requires the name of the example file as an input
parameter. Start it with a command that looks like:

````
bokeh serve --show  /opt/OpendTect/6.6.0/bin/python/dgbpy/bokeh_apps/example_viewer --args '{"Examples File": "/ODData/F3_Demo_2020/NLAs/test_img2imgreg_data.h5"}'

````
where the location of the example_view folder and the example file location will
depend on your installation. Note the use of both single and double quotes. Proper
escaping of the parameter string is essential to stop the command line shell
breaking it up.

## TrainUI App
The trainui app starts the Bokeh user interface for the machine learning plugin
training. The app requires a number of parameters so the parameter string is
quite complex. Start the app with a command that looks like:

````
bokeh serve --show /opt/OpendTect/6.6.0/bin/python/dgbpy/bokeh_apps/trainui --args
'{"Examples File": "/mnt/Data/seismic/ODData/F3_Demo_2020/NLAs/test_img2imgreg_data.h5",
"Output Model File": "/mnt/Data/seismic/ODData/F3_Demo_2020/NLAs/test_img2imgreg_model2.h5",
"Training Type": "New", "ProcLog File": "/mnt/Data/seismic/ODData/F3_Demo_2020/Proc/test_img2imgreg_model2_log.txt",
"ComArgs": {"dtectdata": "/mnt/Data/seismic/ODData"}}'
````
where the location of the trainui folder and the details within the parameter string
will depend on your installation and workflow.
