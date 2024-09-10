# vserver stop -n beweeghuis
# docker volume prune -f
$server_config = $(Convert-Path -path $(Resolve-Path beweeghuis.yaml))
vserver start -c $server_config
sleep 10
$server_entities = $(Convert-Path -path $(Resolve-Path entities.yaml))
vserver import -c $server_config $server_entities