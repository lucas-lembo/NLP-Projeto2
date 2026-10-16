# Guia de revisão dos pares de teste

Arquivo: `test_pairs_to_annotate.csv` (210 pares dos 56 casos do Projeto 1, 30 por tipo de par).

A coluna `suggested_label` foi pré-preenchida pelo Claude. Cada pessoa revisa **42 linhas**:

| Quem | Linhas da planilha |
|---|---|
| Pessoa 1 | 1–42 |
| Pessoa 2 | 43–84 |
| Pessoa 3 | 85–126 |
| Pessoa 4 | 127–168 |
| Pessoa 5 | 169–210 |

## Como revisar

1. Leia a `sentence`. As entidades estão marcadas como `[E1 ...]` e `[E2 ...]`.
2. Se concordar com a sugestão, copie-a para `gold_label`. Se não concordar, escreva o rótulo certo. Ele precisa estar em `allowed_labels`.
3. Preencha `reviewer` com seu nome. Use `notes` se quiser justificar.

Não deixe `gold_label` vazio. Enquanto houver linha vazia, a avaliação usa a sugestão e avisa.

## O que cada rótulo significa

A pergunta é sempre: **a frase afirma essa relação para este paciente?** O que vale é o que o texto diz, não o que sabemos de medicina.

| Rótulo | Par | Quando usar | Exemplo |
|---|---|---|---|
| TREATED_BY | diagnóstico → tratamento | O tratamento foi dado (ou indicado) para aquele diagnóstico | "[E1 herpes zoster] and treated with [E2 acyclovir]" |
| SUPPORTS | sintoma → diagnóstico | O sintoma é manifestação ou evidência do diagnóstico | "[E1 hypoxemic] because of aspiration [E2 pneumonia]" |
| LOCATED_IN | diagnóstico → local | O diagnóstico está naquele local do corpo | "[E1 synovitis] in bilateral metacarpophalangeal [E2 joints]" |
| TARGETS | tratamento → local | O tratamento é aplicado naquele local | "[E2 tracheal] [E1 intubation]" |
| PERFORMED_ON | exame → local | O exame foi feito naquele local | "[E1 tomography] of the left [E2 upper extremity]" |
| CONFIRMS | exame → diag./sintoma | O exame confirma o achado, ou é compatível com ele ("confirmed", "consistent with", "diagnosed by") | "[E1 biopsy] ... consistent with [E2 adenocarcinoma]" |
| REVEALS | exame → diag./sintoma | O exame mostrou o achado ("showed", "revealed") | "[E1 colonoscopy] showed extensive [E2 colitis]" |
| EXCLUDES | exame → diag./sintoma | O exame descartou o achado ("no evidence of", "did not show") | "[E1 biopsy] showed no features of [E2 myocarditis]" |
| NONE | qualquer | Nenhuma das anteriores | |

## Casos de dúvida (decisões já tomadas na pré-anotação)

- **Erro do NER:** se uma entidade tem o tipo errado (ex.: "aspirin" marcada como exame), use `NONE`. A relação não pode estar certa se a entidade está errada.
- **Exame só planejado ou hipotético** ("was recommended", "to exclude"): `NONE`.
- **Mesma palavra repetida:** vale a menção marcada. Em "serum sodium ..., serum glucose", o par `[E1 sodium]`–`[E2 serum]` só é `PERFORMED_ON` se for o "serum" logo antes de "sodium".
- **Listas de exames ou diagnósticos** ("negative for A, B and C"): itens da mesma lista não se relacionam entre si (`NONE`).
- **Ida para a UTI por causa da doença** ("necessitating transfer to the intensive care unit"): marcamos `TREATED_BY`. É um caso ambíguo, indicado em `notes`.
