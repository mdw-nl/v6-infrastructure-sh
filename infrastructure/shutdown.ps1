echo "please use the bash shell script"
exit

.\venv\Scripts\activate
vnode stop -n ortho
vnode stop -n fysio
vserver stop -n beweeghuis
deactivate
Remove-Item -Recurse -Force C:\ProgramData\vantage6\server\beweeghuis*

docker volume rm $(docker volume ls -f name=vantage6 -q)
docker volume rm $(docker volume ls -f name=ortho -q)
docker volume rm $(docker volume ls -f name=fysio -q)