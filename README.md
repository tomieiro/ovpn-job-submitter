# ovpn-job-submitter

Envia um notebook Jupyter ao cluster SLURM do C4AI pela VPN da USP,
acompanha a execução e baixa o `.ipynb` completo, com todas as células
executadas.

## 1. Instale a biblioteca

Python 3.10 ou mais recente é necessário. Dentro deste repositório, execute:

```bash
python -m pip install -e '.[runner]'
```

No Windows, use `py` no lugar de `python` se necessário.

## 2. Instale o OpenVPN

A biblioteca identifica Linux, Windows e macOS automaticamente. Basta instalar
o OpenVPN uma vez:

- **Windows:** baixe e execute o
  [instalador oficial OpenVPN Community](https://openvpn.net/community/).
  Depois, reabra o terminal.
- **Ubuntu/Debian:** `sudo apt update && sudo apt install openvpn`
- **Fedora:** `sudo dnf install openvpn`
- **Arch/Manjaro:** `sudo pacman -S openvpn`
- **Outras distribuições Linux:** consulte os
  [repositórios oficiais do OpenVPN](https://community.openvpn.net/Pages/OpenVPN%20software%20repos).
- **macOS:** instale o [Homebrew](https://brew.sh/) e execute
  `brew install openvpn`. Veja também a
  [página oficial do pacote](https://formulae.brew.sh/formula/openvpn).

No Linux e macOS, a biblioteca pede a senha do `sudo` quando precisa abrir a
VPN. No Windows, abra o PowerShell ou Terminal com **Executar como
administrador** antes de rodar o script.

Também é possível conectar à VPN da USP manualmente pelo aplicativo gráfico.
Se o servidor SSH já estiver acessível, a biblioteca reutiliza a conexão e não
a encerra no final.

O servidor precisa estar salvo em `~/.ssh/known_hosts`. Para fazer isso,
conecte uma vez com `ssh <usuario>@<servidor>`, confira a identificação
apresentada e aceite a chave.

## 3. Organize os arquivos

Use esta estrutura:

```text
workspace/
├── SSH/
│   ├── c4ai.icmc.usp.br.ovpn
│   ├── ca-cert.pem
│   ├── client-<usuario>-cert.pem
│   └── client-<usuario>-key.pem
└── project/
    ├── experimento.ipynb
    └── dados.nc
```

A pasta indicada por `vpn_dir` deve conter exatamente um `.ovpn` e os
certificados referenciados por ele.

## 4. Rode o notebook

Este é o script completo:

```python
from dgx_slurm import run_notebook

run_notebook(
    "project/experimento.ipynb",
    include_project_files=True,
    vpn_dir="SSH",
    ssh_host="c4aiscm2",
    ssh_port=22,
    partition="devwork",
    gpus=1,
    cpus=8,
    memory="0",
    time_limit="04:00:00",
)
```

Altere o caminho do notebook e use `include_project_files=False` quando não
quiser enviar os outros arquivos da pasta `project`.

O diretório da VPN, endereço SSH e todos os recursos do SLURM são obrigatórios.
A biblioteca não escolhe silenciosamente infraestrutura ou custos de
processamento.

Durante a execução, ela:

1. valida o `.ovpn` e identifica o usuário pelo certificado;
2. localiza o OpenVPN adequado ao sistema e abre a VPN, se necessário;
3. envia o notebook e os dados solicitados;
4. submete e acompanha o job no SLURM;
5. salva o resultado como `experimento.executed.ipynb`;
6. fecha somente as conexões que ela própria abriu.

Ambientes virtuais, Git, resultados anteriores e credenciais nunca são
incluídos no envio.

As dependências específicas do projeto devem ser instaladas no próprio
notebook ou incluídas na imagem do container. Exemplo:

```python
%pip install -q xarray netcdf4 cartopy
```

## Opções adicionais

O usuário inferido do certificado e o caminho de saída podem ser substituídos:

```python
result = run_notebook(
    "project/experimento.ipynb",
    include_project_files=True,
    vpn_dir="/caminho/para/SSH",
    ssh_host="outro-servidor",
    ssh_port=22,
    partition="research",
    gpus=2,
    cpus=16,
    memory="128G",
    time_limit="02:00:00",
    username="usuario-do-cluster",
    output="resultado.ipynb",
)
print(result.executed_notebook)
```

Para controlar manualmente o ciclo de vida, use `DGXClient.submit()` e
`await DGXJob.wait()`.

## Testes

```bash
python -m pytest -m "not cluster"
```
