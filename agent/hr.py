# 写一套智能体流程来模拟企业招人，一个agent为人事角色，评价当前简历，另一个agent负责业务评价，如果通过就由人事agent给出offer

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
import re

def extract_xml(text: str, tag: str) -> str:
    """
    Extracts the content of the specified XML tag from the given text. Used for parsing structured responses 

    Args:
        text (str): The text containing the XML.
        tag (str): The XML tag to extract content from.

    Returns:
        str: The content of the specified XML tag, or an empty string if the tag is not found.
    """
    match = re.search(f'<{tag}>(.*?)</{tag}>', text, re.DOTALL)
    return match.group(1) if match else ""

if __name__ == "__main__":
# 人事agent评价简历，是否能够交给业务评价

    #流程
    #1.对话，获取用户简历信息
    llm=ChatOpenAI(model='gpt-4o',
                         temperature=0.05, api_key="skxx")
    #初始化大模型，给出提示词，要求给出固定格式的评价
    hr_prompt="""
    你是一个人事经理，现在需要你分析将处理的事情：
    1.评价一个应聘者的简历，请给出评价并给出决策，是否能够交给业务评价。能就返回True，不能就返回False。不要返回其他内容。
    2.收到业务经理的评价，请给出决策，是否能够给出offer。能就返回True，不能就返回False。不要返回其他内容。
    请按照以下格式给出评价：
    ```xml
    <analysis>
    解释你的分析过程和任务拆解过程。
    </analysis>

    <decision>
    给出你的决策，是否能够交给业务评价。能就返回True，不能就返回False。不要返回其他内容。
    </decision>
    ```
    """

    business_prompt="""
    你是一个开发岗业务经理，现在需要评价一个应聘者的简历，请给出评价。
    请按照以下格式给出评价：
    ```xml
    <analysis>
    解释你的分析过程和任务拆解过程。
    </analysis>

    <decision>
    给出你的决策，是否能够交给人事。
    </decision>
    ```
    """


    # 定义模板
    hr_template = ChatPromptTemplate.from_messages([
        ("system", hr_prompt),
        ("human", "{resume}")  # 使用命名参数
    ])

    business_template = ChatPromptTemplate.from_messages([
        ("system", business_prompt),
        ("human", "{resume}"),  # 使用命名参数
        ("human", "{hr_analysis}")
    ])

    # 创建chain
    chain1 = hr_template | llm

    chain2 = business_template | llm
    # 获取用户简历
    resume1="""
# 个人简历

## 基本信息
- **姓名**：张明
- **年龄**：28岁
- **学历**：硕士研究生
- **专业**：计算机科学与技术
- **邮箱**：zhangming@email.com

## 教育背景
### 北京大学（2018-2021）
- 计算机科学与技术，硕士学位
- GPA：3.8/4.0
- 主修课程：机器学习、深度学习、计算机视觉

### 浙江大学（2014-2018）
- 软件工程，学士学位
- GPA：3.7/4.0

## 工作经验
### 腾讯科技（2021-至今）
**高级算法工程师**
- 负责推荐系统核心算法开发和优化
- 设计并实现了基于深度学习的用户行为预测模型
- 带领团队完成了广告系统的技术改造，提升收益30%

### 字节跳动（2020-2021）
**算法实习生**
- 参与短视频推荐算法研发
- 优化特征工程流程，提升模型效果

## 专业技能
- 编程语言：Python, Java, C++
- 机器学习框架：PyTorch, TensorFlow
- 数据库：MySQL, MongoDB
- 开发工具：Git, Docker, Linux

## 项目经验
### 智能推荐系统优化项目
- 设计并实现多目标优化框架
- 使用强化学习提升用户长期留存
- 项目取得超过15%的效果提升

## 获奖经历
- 2020年 ACM国际算法竞赛银奖
- 2019年 研究生国家奖学金
    """
    resume2="""
# 个人简历

## 基本信息
- **姓名**：李小明
- **年龄**：25岁 
- **学历**：大专
- **专业**：计算机应用
- **邮箱**：lixm@qq.com

## 教育背景
### 某职业技术学院（2017-2020）
- 计算机应用技术专业
- 成绩一般

## 工作经验
### 某网络公司（2020-2021）
**初级程序员**
- 负责简单的网页制作和维护
- 经常迟到早退,工作态度消极
- 因表现不佳被辞退

### 某培训机构（2021-2022） 
**助教**
- 辅助老师整理教学资料
- 工作期间频繁请假
- 半年后主动离职

## 专业技能
- 会基础HTML和CSS
- 会用Word和Excel
- 英语水平较差

## 项目经验
### 公司网站维护
- 修改网站文字内容
- 上传产品图片
- 工作中出现多次错误

## 其他说明
- 无专业证书
- 无获奖经历
- 工作经历不稳定
- 缺乏团队合作精神

    """
    #2.调用llm
    # 调用时使用字典格式
    hr_response = chain1.invoke({
        "resume": resume2
    })
    # 获取决策结果
    decision = extract_xml(hr_response.content, "decision").strip().lower() == "true"
    analysis = extract_xml(hr_response.content, "analysis")
    print("\n是否递交给业务岗位:",decision,"\n原因是：",analysis)

    if decision:
        print("递交给业务岗位")
        business_response = chain2.invoke({
            "resume": resume2,
            "hr_analysis": analysis
        })

        business_decision = extract_xml(business_response.content, "decision")
        business_analysis = extract_xml(business_response.content, "analysis")
        print("\n业务决策:",business_decision,"\n原因是：",business_analysis)

        hr_response = chain1.invoke({
        "business_decision": business_decision
        })

        hr_decision = extract_xml(hr_response.content, "decision")
        print("\n人事offer决策:",hr_decision)
    else:
        print("拒绝offer")








