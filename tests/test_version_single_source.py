#!/usr/bin/env python3
"""
Una sola fonte di verita' per la versione del pacchetto.

Fino alla v1.3.5 la versione viveva in DUE posti: `dbsi_toolbox/__init__.py` e
`setup.py`. Alla 1.3.5 hanno divergito -- il sorgente diceva 1.3.5 e
`pip install -e .` registrava 1.3.4 -- e prima ancora i metadati installati
riportavano 3.0.0 per lo stesso albero.

In editable install la divergenza e' **solo un'etichetta**: `dbsi_toolbox.__version__`
si legge dall'albero dei sorgenti, quindi le guardie di versione nei notebook
vedono il numero vero e il fit che gira e' quello giusto. Ma in un'installazione
non-editable si spedirebbe il numero sbagliato, e la provenienza scritta nel
run report vale quanto vale la stringa di versione.

    python tests/test_version_single_source.py
"""
import re
import sys
from pathlib import Path

RADICE = Path(__file__).resolve().parent.parent


def test_setup_non_ha_la_versione_scritta_a_mano():
    print("\n=== test_setup_non_ha_la_versione_scritta_a_mano ===")
    src = (RADICE / 'setup.py').read_text(encoding='utf-8')
    # cerca `version="1.2.3"` dentro la chiamata a setup()
    letterali = re.findall(r'version\s*=\s*["\'](\d+\.\d+[^"\']*)["\']', src)
    print(f"  letterali di versione in setup.py: {letterali or 'nessuno'}")
    assert not letterali, (
        f"setup.py contiene ancora una versione scritta a mano {letterali}: "
        "deve derivarla da dbsi_toolbox/__init__.py, altrimenti i due numeri "
        "torneranno a divergere al prossimo rilascio")
    assert '_version()' in src, "setup.py non chiama piu' _version()"
    print("  [ok]")


def test_setup_e_pacchetto_concordano():
    print("\n=== test_setup_e_pacchetto_concordano ===")
    init = (RADICE / 'dbsi_toolbox' / '__init__.py').read_text(encoding='utf-8')
    m = re.search(r'^__version__\s*=\s*["\']([^"\']+)["\']', init, re.M)
    assert m, "__version__ non trovata in dbsi_toolbox/__init__.py"
    dal_sorgente = m.group(1)

    sys.path.insert(0, str(RADICE))
    import dbsi_toolbox
    print(f"  __init__.py            {dal_sorgente}")
    print(f"  dbsi_toolbox.__version__ {dbsi_toolbox.__version__}")
    assert dbsi_toolbox.__version__ == dal_sorgente
    # e setup.py deve estrarre esattamente quella
    sys.path.insert(0, str(RADICE))
    src = (RADICE / 'setup.py').read_text(encoding='utf-8')
    ns = {'__file__': str(RADICE / 'setup.py')}
    corpo = src[src.index('def _version()'):src.index('\n\n\nsetup(')]
    exec('import re\nfrom pathlib import Path\n' + corpo, ns)
    dal_setup = ns['_version']()
    print(f"  setup.py:_version()    {dal_setup}")
    assert dal_setup == dal_sorgente, (
        f"setup.py estrae {dal_setup} ma il pacchetto dice {dal_sorgente}")
    print("  [ok]")


def test_metadati_installati():
    """AVVISO, non errore: in editable i metadati si aggiornano solo al reinstall.

    Un disallineamento subito dopo un bump di versione e' normale e NON e' un
    difetto del codice -- il fit che gira e' quello dell'albero. Serve solo a
    ricordare di rifare `pip install -e .` prima di fidarsi dei metadati.
    """
    print("\n=== test_metadati_installati (solo avviso) ===")
    sys.path.insert(0, str(RADICE))
    import dbsi_toolbox
    try:
        import importlib.metadata as md
        installata = md.version('dbsi-toolbox')
    except Exception as e:
        print(f"  metadati non disponibili ({e}) -- nulla da controllare")
        print("  [ok]")
        return
    print(f"  albero dei sorgenti {dbsi_toolbox.__version__}   "
          f"metadati pip {installata}")
    if installata != dbsi_toolbox.__version__:
        print(f"  [AVVISO] i metadati pip dicono {installata} ma il codice e' "
              f"{dbsi_toolbox.__version__}.")
        print(f"           In editable install e' solo un'etichetta: le guardie "
              f"dei notebook leggono {dbsi_toolbox.__version__} e il fit e' "
              f"quello giusto.")
        print(f"           Per allinearle: pip install -e .")
    else:
        print("  allineati")
    print("  [ok]")


if __name__ == '__main__':
    for f in (test_setup_non_ha_la_versione_scritta_a_mano,
              test_setup_e_pacchetto_concordano,
              test_metadati_installati):
        try:
            f()
        except AssertionError as e:
            print(f"\n!!! {f.__name__}: {e}")
            sys.exit(1)
    print("\ntutti i test passati")
