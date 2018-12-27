1. Build DockFile by command line : docker build -t yong/climate_analysis:latest .
2. Run a container by command line: 
nvidia-docker run --name climate_analysis -it -v 'your workspace':/notebooks/workspace/ -p 8888:8888  -p 6006:6006 yong/climate_analysis:latest
3. In your browser, open the URL http://localhost:8888/. All notebooks from your session will be saved in 'your workspace'.
4. Go into the container by command line: docker exec -it climate_analysis bash
5. python /notebooks/samples/setup.py develop
6. Change the keras config file as: 
{
    "epsilon": 1e-07,
    "floatx": "float32",
    "image_data_format": "channels_first",
    "backend": "tensorflow"
}
7. Run experiments by : python /notebooks/Climate_Analysis/Yong/python experiment.py -data 10 -forecast 0 -model convlstm -layers 1 -filters 40 -kernel s -norm 1 -gpu 2 -lr 0.0001
