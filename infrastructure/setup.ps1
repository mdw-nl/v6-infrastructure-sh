echo "please use the bash shell script"
exit

Set-Variable -Name "VERSION_NODE" -Value "3.8.2"
Set-Variable -Name "VERSION_SERVER" -Value $VERSION_NODE

python -m venv .\venv
.\venv\Scripts\activate

pip install -r requirements.txt

# Technically this is not needed, but makes life more easy ;-)
#docker pull harbor2.vantage6.ai/infrastructure/server:petronas
# docker pull harbor2.vantage6.ai/infrastructure/server:3.2.0
#docker pull harbor2.vantage6.ai/infrastructure/node:petronas
# docker pull harbor2.vantage6.ai/infrastructure/node:3.2.0

# Start server
$server_config = $(Convert-Path -path $(Resolve-Path beweeghuis.yaml))
vserver start -c $server_config --image harbor2.vantage6.ai/infrastructure/server:$VERSION_SERVER

# Import server entities
$server_entities = $(Convert-Path -path $(Resolve-Path entities.yaml))
vserver import -c $server_config $server_entities

# Fysio node start
$fysio_config = $(Convert-Path -path $(Resolve-Path fysio.yaml))
# vnode create-private-key -c $fysio_config -e application --overwrite
vnode start -c $fysio_config --image harbor2.vantage6.ai/infrastructure/node:$VERSION_NODE

# Ortho node start
$ortho_config = $(Convert-Path -path $(Resolve-Path ortho.yaml))
# vnode create-private-key -c $ortho_config -e application --overwrite
vnode start -c $ortho_config --image harbor2.vantage6.ai/infrastructure/node:$VERSION_NODE