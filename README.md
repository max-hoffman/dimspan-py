# Dimension Expansion with SINDy (Python)
The makefile and setup.py's are still boilerplate, need to fix.

## local dev with docker
1. download docker client
2. make sure client is running
3. go to the current folder
4. build docker image from local directory ("./", and tag with name "dimspan"
```
docker build -t dimspan .
```
5. run the docker image "dimspan" interactively (so you can see console output), and remove the container upon completion
```
docker run --rm dimspan
```

## local dev with python
1. virtualenv if you want to
2. install requirements
```
pip install -r requirements.txt
```
3. run main
```
python dimspan/main.py
```