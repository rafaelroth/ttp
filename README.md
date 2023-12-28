#Solução de Simulated Annealing para Traveling Tournament Problem - TTP

O Traveling Tournament Problem - TTP, é um problema de otimização combinatória complexo e desafiador, comum na organização de torneios esportivos, especialmente em campeonatos profissionais. O objetivo central do TTP é elaborar um cronograma de torneio equilibrado, onde cada equipe enfrenta todas as outras, uma vez em casa e uma vez fora, buscando minimizar a distância total percorrida por todas as equipes.

## Metodologia Utilizada: Simulated Annealing
A meta-heurística Simulated Annealing (SA) foi escolhida para resolver o TTP. Inspirada no processo de recozimento dos metais, SA é uma técnica eficaz para explorar o espaço de soluções, aceitando soluções subótimas inicialmente para evitar mínimos locais e, gradualmente, refinando a busca em direção a uma solução ótima ou próxima.

## Implementação em Python
O projeto foi implementado em Python, uma linguagem de programação reconhecida pela sua versatilidade e aplicabilidade em algoritmos de otimização e análise de dados. Python fornece acesso a uma ampla gama de bibliotecas que facilitam o manuseio de dados e a implementação de algoritmos complexos.

## Aplicação: Campeonato Brasileiro de 2023
O algoritmo foi adaptado para o Campeonato Brasileiro de 2023, considerando as localizações reais das equipes participantes. Utilizamos o Simulated Annealing para desenvolver um cronograma de jogos que minimiza a distância total percorrida pelas equipes. Uma adaptação importante feita foi a regra do torneio, que permite um cronograma não espelhado, ou seja, os jogos do turno e returno não precisam ocorrer na mesma ordem, proporcionando uma flexibilidade maior na programação dos jogos e atendendo a critérios logísticos e de equidade.

## Resultados

Comparando os calendários de jogos gerados pelo algoritmo com o formato atual do campeonato, observamos uma diminuição significativa nas distâncias de viagem para todas as equipes, o que sugere uma maior justiça competitiva e redução de custos logísticos. Além disso, discutimos como as adaptações de formato propostas complementam os resultados obtidos pelo algoritmo, oferecendo uma solução integrada para o problema de agendamento.


## Tabela 1. Distância por Time sem Otimização
| Time         | Distância Percorrida (km) |
|--------------|---------------------------|
| América-MG   | 49876.77                  |
| Athletico-PR | 48674.44                  |
| Atlético-MG  | 48146.87                  |
| Bahia        | 70348.85                  |
| Botafogo     | 48797.95                  |
| Bragantino   | 47815.85                  |
| Corinthians  | 47593.83                  |
| Coritiba     | 50684.74                  |
| Cruzeiro     | 42956.13                  |
| Cuiabá       | 70534.81                  |
| Flamengo     | 50686.03                  |
| Fluminense   | 45764.64                  |
| Fortaleza    | 98274.22                  |
| Goiás        | 55502.79                  |
| Grêmio       | 61781.33                  |
| Internacional| 62058.85                  |
| Palmeiras    | 44095.43                  |
| Santos       | 44796.44                  |
| São Paulo    | 44522.02                  |
| Vasco da Gama| 48783.26                  |


## Tabela 2. Distância por Time após Otimização
| Time         | Distância Percorrida (km) |
|--------------|---------------------------|
| América-MG   | 27355.58                  |
| Athletico-PR | 28328.51                  |
| Atlético-MG  | 29547.69                  |
| Bahia        | 40442.44                  |
| Botafogo     | 23823.48                  |
| Corinthians  | 24388.54                  |
| Coritiba     | 33473.39                  |
| Cruzeiro     | 31449.69                  |
| Cuiabá       | 44461.57                  |
| Flamengo     | 25383.86                  |
| Fluminense   | 22984.39                  |
| Fortaleza    | 59077.35                  |
| Goiás        | 31929.67                  |
| Grêmio       | 34744.7                   |
| Internacional| 40909.42                  |
| Palmeiras    | 22736.47                  |
| Bragantino   | 25838.27                  |
| Santos       | 26927.96                  |
| São Paulo    | 21379.55                  |
| Vasco da Gama| 24542.41                  |


- Fortaleza: Uma das maiores reduções observadas foi para o Fortaleza, que teve sua distância de viagem reduzida de 98.274,22 km para 59.077,35 km. Isso representa uma diminuição de quase 40% na distância percorrida, um resultado impressionante considerando as grandes distâncias geográficas no Brasil.
- Cuiabá e Bahia: Ambos também apresentaram reduções significativas, com o Cuiabá passando de 70.534,81 km para 44.461,57 km e o Bahia de 70.348,85 km para 40.442,44 km. Estas reduções são particularmente notáveis, pois demonstram a capacidade do algoritmo de ajustar eficientemente as rotas mesmo em regiões com desafios geográficos específicos.
- Reduções Consistentes: Todos os times mostraram reduções consideráveis nas distâncias percorridas. Por exemplo, o Flamengo teve sua distância reduzida de 50.686,03 km para 25.383,86 km, enquanto o Palmeiras passou de 44.095,43 km para 22.736,47 km. Esses resultados enfatizam a eficácia do Simulated Annealing em distribuir os jogos de maneira a minimizar a distância total percorrida.
