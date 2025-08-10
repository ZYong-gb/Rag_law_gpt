"""
加载示例法律数据
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy.orm import Session
from app.database.connection import db_manager
from app.models.schemas import DocumentCreate
from app.services.document_processor import document_processor


def create_sample_documents():
    """创建示例法律文档"""
    
    sample_docs = [
        {
            "title": "中华人民共和国民法典 - 总则编",
            "content": """
第一条 为了保护民事主体的合法权益，调整民事关系，维护社会和经济秩序，适应中国特色社会主义发展要求，弘扬社会主义核心价值观，根据宪法，制定本法。

第二条 民法调整平等主体的自然人、法人和非法人组织之间的人身关系和财产关系。

第三条 民事主体的人身权利、财产权利以及其他合法权益受法律保护，任何组织或者个人不得侵犯。

第四条 民事主体在民事活动中的法律地位一律平等。

第五条 民事主体从事民事活动，应当遵循自愿原则，按照自己的意思设立、变更、终止民事法律关系。

第六条 民事主体从事民事活动，应当遵循公平原则，合理确定各方的权利和义务。

第七条 民事主体从事民事活动，应当遵循诚信原则，秉持诚实，恪守承诺。

第八条 民事主体从事民事活动，不得违反法律，不得违背公序良俗。
            """,
            "category": "民法",
            "source": "《中华人民共和国民法典》"
        },
        {
            "title": "合同法基本原则",
            "content": """
第四百六十四条 合同是民事主体之间设立、变更、终止民事法律关系的协议。

第四百六十五条 依法成立的合同，受法律保护。
依法成立的合同，仅对当事人具有法律约束力，但是法律另有规定的除外。

第四百六十六条 当事人对合同条款的理解有争议的，应当依据本法第一百四十二条第一款的规定，确定争议条款的含义。
合同文本采用两种以上文字订立并约定具有同等效力的，对各文本使用的词句推定具有相同含义。各文本使用的词句不一致的，应当根据合同的相关条款、性质、目的以及诚信原则等予以解释。

第四百六十七条 本法或者其他法律没有明文规定的合同，适用本编通则的规定，并可以参照适用本编或者其他法律最相类似合同的规定。
            """,
            "category": "民法",
            "source": "《中华人民共和国民法典》"
        },
        {
            "title": "劳动合同法基本规定",
            "content": """
第一条 为了完善劳动合同制度，明确劳动合同双方当事人的权利和义务，保护劳动者的合法权益，构建和发展和谐稳定的劳动关系，制定本法。

第二条 中华人民共和国境内的企业、个体经济组织、民办非企业单位等组织（以下称用人单位）与劳动者建立劳动关系，订立、履行、变更、解除或者终止劳动合同，适用本法。

第三条 订立劳动合同，应当遵循合法、公平、平等自愿、协商一致、诚实信用的原则。
依法订立的劳动合同具有约束力，用人单位与劳动者应当履行劳动合同约定的义务。

第四条 用人单位应当依法建立和完善劳动规章制度，保障劳动者享有劳动权利、履行劳动义务。
            """,
            "category": "劳动法",
            "source": "《中华人民共和国劳动合同法》"
        }
    ]
    
    return sample_docs


def load_sample_data():
    """加载示例数据到系统"""
    try:
        # 初始化数据库
        db_manager.init_database()
        
        # 获取数据库会话
        db = db_manager.get_db_session()
        
        try:
            # 创建示例文档
            sample_docs = create_sample_documents()
            
            for doc_data in sample_docs:
                # 创建临时文件
                temp_file = f"./data/temp_{doc_data['title'][:10]}.txt"
                
                with open(temp_file, 'w', encoding='utf-8') as f:
                    f.write(doc_data['content'])
                
                # 创建文档对象
                document_create = DocumentCreate(
                    title=doc_data['title'],
                    content=doc_data['content'],
                    category=doc_data['category'],
                    source=doc_data['source'],
                    file_path=temp_file,
                    file_type='.txt'
                )
                
                # 处理文档
                try:
                    doc_id = document_processor.process_document(
                        temp_file, 
                        document_create, 
                        db
                    )
                    print(f"✓ 加载文档: {doc_data['title']} (ID: {doc_id})")
                    
                except Exception as e:
                    print(f"✗ 加载文档失败 {doc_data['title']}: {e}")
                
                # 清理临时文件
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            
            print(f"\n✅ 示例数据加载完成!")
            
        finally:
            db.close()
            
    except Exception as e:
        print(f"✗ 示例数据加载失败: {e}")


if __name__ == "__main__":
    print("🔄 开始加载示例法律数据...")
    load_sample_data()
