"""Build the verified 78-page Vietnamese report from source and real results."""
from expanded_content import (front,introduction,theory,datasets,model_design,
    numpy_implementation,pytorch_implementation,tensorflow_implementation,
    results,conclusion,references,appendices,fill_indexes)
from report_engine import ROOT,PAGES,FIGURES,TABLES,CODES,render

def main():
    front();introduction();theory();datasets();model_design()
    numpy_implementation();pytorch_implementation();tensorflow_implementation()
    results();conclusion();references();appendices();fill_indexes()
    print('Content:',len(PAGES)+1,'pages;',len(FIGURES),'figures;',len(TABLES),'tables;',len(CODES),'code listings')
    render(ROOT/'report/Assignment04_NguyenNgocHoangNam_B23DCCN585.pdf')

if __name__=='__main__':main()
