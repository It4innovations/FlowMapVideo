# flowmapopt

* [Roadmap](../../wikis/Roadmap)

## Zadání
**Téma**: Optimalizace vizualizace vývoje dopravního toku v čase.

Student nebo studentka se bude v rámci své práce věnovat optimalizaci vizualizace vývoje dopravního toku v čase. Informace o dopravě budou získávány z dopravního simulátoru, který pro daný časový interval, a tak zvanou origin-destination matici, provede simulaci průběhu jednotlivých vozidel na mapě. Pohyby jednotlivých vozidel jsou v pravidelných intervalech zaznamenávány. Cílem práce je vytvořit vizualizaci relativně velké simulace (jednotky až desítky tisíc vozidel) v rozumném čase (desítky minut až jednotky hodin).

Jednotlivé body zadání jsou:
1.	Seznámit se s aktuálním stavem dopravního simulátoru vyvíjeném na IT4innovations, zejména s formátem záznamu pohybu aut.
2.	Prozkoumat možnosti použitého vizualizačního nástroje a analyzovat problémová místa z pohledu výkonu.
3.	Navrhnout řešení která povedou ke zlepšení výkonu vizualizace.
4.	Implementovat navržená řešení ať už změnami ve vizualizačním nástroji, případně změnami v použitých knihovnách.

## Literatura

* https://matplotlib.org/
* https://d3js.org/
* W. Chen, F. Guo and F. -Y. Wang, "A Survey of Traffic Data Visualization," in IEEE Transactions on Intelligent Transportation Systems, vol. 16, no. 6, pp. 2970-2984, Dec. 2015, doi: 10.1109/TITS.2015.2436897.
  * [link](https://ieeexplore.ieee.org/abstract/document/7120975?casa_token=SS_93qCqCkoAAAAA:HoxHGaz1nd4d4u_TCP7qhNqVbFyGSFSGeUl7hip1F0jfK0h17_CniYEfNoPmTdoi5fMxwAkiBnA)
* D. Guo, "Flow Mapping and Multivariate Visualization of Large Spatial Interaction Data," in IEEE Transactions on Visualization and Computer Graphics, vol. 15, no. 6, pp. 1041-1048, Nov.-Dec. 2009, doi: 10.1109/TVCG.2009.143.
  * [link](https://ieeexplore.ieee.org/abstract/document/5290710?casa_token=DC5BcbHVqmoAAAAA:R_xmPvuwNwrCzqfsHh90M3khegg0MxahsWsKT1UN7WGlf7LunyogQldebv8ZvKYtZWyi8h5UY9I)


## Instalace

### Prerekvizity

Pro uložení videa je potřeba `FFmpeg`.

##

1. Vytvoření a aktivace virtuálnáho prostředí:
```
virtualenv <VENV>
source <VENV>/bin/activate
```

2. Instalace přes pip
```
python3 -m pip install git+https://project_1327_bot:glpat-YdTTaGALriUSyz7DDmix@code.it4i.cz/intern/trafficflowmap/flowmapopt.git@dev 
```

## Spuštění
```
traffic-flow-map --help
```
* DATA_FILE
  * cesta k PARQUET souboru se sloupci:
    * timestamp
    * node_from (osmnx id)
    * node_to (osmnx id)
    * vehicle_id
    * start_offset_m 
  * v případě použití možnosti ```-p```/```-processed-data``` cesta k CSV souboru se sloupci:
    * timestamp
    * node_from (osmnx id)
    * node_to (osmnx id)
    * count_from (počet vozidel v první polovině cesty)
    * count_to (počet vozidel ve druhé polovině cesty)
* MAP_FILE - cesta k GRAPHML souboru


