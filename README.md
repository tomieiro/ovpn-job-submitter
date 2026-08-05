# ovpn-job-submitter

Envia um notebook Jupyter ao cluster SLURM do C4AI pela VPN da USP,
acompanha a execução e baixa o `.ipynb` completo, com todas as células
executadas.

## 1. Baixe o programa

Baixe a versão mais recente em
[GitHub Releases](https://github.com/tomieiro/ovpn-job-submitter/releases/latest):

- **Windows (janela):** `ovpn-job-submitter-windows-x86_64-gui.exe`
- **Windows (linha de comando):** `ovpn-job-submitter-windows-x86_64.exe`
- **Linux:** `ovpn-job-submitter-linux-x86_64`
- **Mac Apple Silicon (M1 ou mais recente):**
  `ovpn-job-submitter-macos-arm64.pkg`
- **Mac Intel:** `ovpn-job-submitter-macos-x86_64.pkg`

O executável já contém o Python e todas as dependências da biblioteca.

No Linux, permita a execução e, opcionalmente, instale o comando:

```bash
chmod +x ovpn-job-submitter-linux-x86_64
sudo mv ovpn-job-submitter-linux-x86_64 /usr/local/bin/ovpn-job-submitter
```

No macOS, abra o `.pkg`; ele instala o comando `ovpn-job-submitter` em
`/usr/local/bin`.

Os executáveis ainda não possuem assinatura comercial. Windows e macOS podem
pedir uma confirmação adicional antes da primeira execução.

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
administrador** antes de rodar o script. Sem isso, o OpenVPN não consegue
configurar o adaptador e fica repetindo
`NETSH: ... ERROR: command failed: returned error code 1`.

Também é possível conectar à VPN da USP manualmente pelo aplicativo gráfico.
Se o servidor SSH já estiver acessível, a biblioteca reutiliza a conexão e não
a encerra no final.

Na primeira conexão, o programa mostra a identificação do servidor
(`SHA256:...`) e pergunta se ela confere — em uma caixa de diálogo na janela,
ou com `[s/N]` no terminal. Ao aceitar, a chave é salva em `~/.ssh/known_hosts`
e a pergunta não se repete. Confira o valor com a identificação divulgada pelo
cluster antes de aceitar; se preferir, faça isso antes com
`ssh <usuario>@<servidor>`, com a VPN ativa.

Se a chave salva mudar depois, o programa recusa a conexão e não envia nada:
isso pode indicar tanto uma reinstalação do servidor quanto uma tentativa de
interceptação, e precisa ser confirmado com os administradores.

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

### Pela janela (Windows)

Clique com o botão direito em `ovpn-job-submitter-windows-x86_64-gui.exe` e
escolha **Executar como administrador**. A janela pede apenas três coisas:

1. o notebook `.ipynb`;
2. a pasta da VPN (`.ovpn` + certificados);
3. a caixa **Enviar também os outros arquivos da pasta do notebook**, que
   equivale ao `--include-files`.

Ao clicar em **Executar no cluster**, a senha do cluster é pedida uma vez e
todo o andamento aparece no painel de log da própria janela. Se o programa for
aberto sem privilégios de administrador, a janela avisa e oferece o botão
**Reabrir como administrador**.

Os demais valores (partição, GPUs, CPUs, memória, tempo) usam os mesmos padrões
da linha de comando; para alterá-los, use o executável de terminal.

A mesma janela pode ser aberta a partir do repositório, mas o pacote precisa
ser instalado antes (o código fica em `src/`, então rodar de dentro da pasta
não basta):

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\ovpn-job-submitter-gui.exe
```

Para abrir com privilégios, clique com o botão direito em
`.venv\Scripts\ovpn-job-submitter-gui.exe` e escolha **Executar como
administrador** — ou use o botão da própria janela.

### Pela linha de comando

No Linux ou macOS:

```bash
ovpn-job-submitter project/experimento.ipynb SSH --include-files
```

No PowerShell:

```powershell
.\ovpn-job-submitter-windows-x86_64.exe `
  "project\experimento.ipynb" "SSH" --include-files
```

No CMD:

```bat
ovpn-job-submitter-windows-x86_64.exe "project\experimento.ipynb" "SSH" --include-files
```

Omita `--include-files` para enviar somente o notebook. Caminhos que contêm
espaços devem ficar entre aspas.

O notebook e a pasta da VPN são obrigatórios. Os demais valores são opcionais:

```text
--ssh-host c4aiscm2
--ssh-port 22
--partition devwork
--gpus 1
--cpus 8
--memory 0
--time-limit 04:00:00
```

Consulte todas as opções com `ovpn-job-submitter --help`.

Durante a execução, o programa:

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

## Uso como biblioteca Python

Para desenvolvimento ou integração com outro código, Python 3.10 ou mais
recente é necessário. Dentro deste repositório:

```bash
python -m pip install -e '.[runner]'
```

Então use a API:

```python
from dgx_slurm import run_notebook

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

## Releases

Cada tag no formato `v*` executa os testes, gera os quatro artefatos e publica
automaticamente uma GitHub Release. Exemplo para a versão declarada em
`pyproject.toml`:

```bash
git tag v0.3.0
git push origin v0.3.0
```

O workflow também pode ser executado manualmente no GitHub Actions para testar
os builds sem publicar uma Release.

## Desenvolvimento e testes

```bash
python -m pytest -m "not cluster"
```
