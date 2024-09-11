# Sobre o projeto
Esse é um modelo de coleta e análise de atividades e iterações na linha produção através de um modelo de captura de imagens/vídeos. Que tem o objetivo  monitorar o tempo de ociosidade dos colaboradores, fazendo essa análise de eficiência de mão de obra entre atividades que agregam e não agregam valor, possibilitando sugestões de melhorias e agilizando o processo de otimização da linha de produção. OBS: Vale ressaltar que o objetivo deste projeto NÃO é "vigiar" ou "monitorar" quem trabalha ou deixa de trabalhar, como forma de punição, mas sim para um controle de média de tempo que cada atividade leva para ser feita, para um melhor gerenciamento de tempo e economia de dinheiro.


# Etapas realizadas do projeto
Preparação dos dados:

Realizamos a instalação de cameras na fábrica, após isso realizamos a gravação dos videos e em seguida fizemos a coleta desse video. Pegamos o video e separamos por frame de imagens, e com essas imagens fizemos a segmentação através do 
Label Studio (https://labelstud.io/guide/install). Aplicamos o model de segmentação do yoloV8 para a detecção dos "Operadores" segmentados.

# Resultados esperados

Controles mais precisos do tempo de cada atividade exclusivamente.
Controle do tempo de ociosidade.



# Autor
Natã Sando

# Linkedin
www.linkedin.com/in/natã-sando-414235260
