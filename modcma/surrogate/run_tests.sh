
PYTHON=python3.10

if [ "$1" = "install" ];
then
	echo 'Installing'
	$PYTHON -m pip install pytest pytest-xdist 
fi


$PYTHON -m pytest -v --numprocesses=auto

