import uuid
from datetime import datetime
from etf_nhti import calculate_nhti_with_input, get_etf
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import Runnable
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from typing import cast
import chainlit as cl

@tool
def recommend_fund(user_id: str) -> str:
    """Tool for getting recommendation of an ETF product.

    Args:
        user_id (str): The user ID to generate recommendations for.

    Returns:
        str: A string containing the recommendation or an error message.
    """
    try:
        # 입력 검증
        if not user_id or not isinstance(user_id, str):
            return "Error: Invalid user ID. Please provide a valid user ID."

        # NHTI 기반 추천 결과
        result = calculate_nhti_with_input(user_id)
        if not result or "No transactions found" in result:
            return f"Sorry, no transactions found for the provided user ID '{user_id}'."
        
        print(result)
        return result

    except Exception as ex:
        print(f"Unexpected error in recommend_fund: {ex}")
        return "Error: An unexpected issue occurred while processing your recommendation. Please try again!"


@tool
def get_etf_detail(etf_ticker: str) -> str:
    """Retrieve detailed information about the composition of an ETF.

    Args:
        etf_ticker (str): The ticker symbol of the ETF.

    Returns:
        str: A formatted string with the ETF components or an error message.
    """
    try:
        # 입력 검증
        if not etf_ticker or not isinstance(etf_ticker, str):
            return "Error: Invalid ETF ticker. Please provide a valid ticker."

        # ETF 구성 요소 정보 가져오기
        etf_data = get_etf(etf_ticker)

        # get_etf 함수가 에러 메시지를 반환한 경우 처리
        if isinstance(etf_data, str):
            return etf_data

        # ETF 구성 요소 데이터 확인
        if etf_data.empty:
            return f"Error: No components found for ETF '{etf_ticker}'. Please check the ticker and try again."

        # 필수 컬럼 검증
        required_columns = {'Stock_Item_Code', 'Sector_Name_English_Y', 'Website_Address', 'Ex_Dividend_Date', 'Dividend_Amount_Per_Stock', 'Dividend_Payment_Currency_Code'}
        if not required_columns.issubset(etf_data.columns):
            return f"Error: ETF data for '{etf_ticker}' is incomplete. Missing required columns."

        # ETF 구성 요소 포맷팅
        formatted_result = f"### ETF '{etf_ticker}' Components:\n"
        for _, row in etf_data.iterrows():
            formatted_result += (
                f"- **Stock_Item_Code**: {row['Stock_Item_Code']}, "
                f"**Website_Address**: {row['Website_Address']}, "
                f"**Sector_Name_English_Y**: {row['Sector_Name_English_Y']}\n"
                f"**Ex_Dividend_Date**: {row['Ex_Dividend_Date']}, "
                f"**Dividend_Amount_Per_Stock**: {row['Dividend_Amount_Per_Stock']}, "
                f"**Dividend_Payment_Currency_Code**: {row['Dividend_Payment_Currency_Code']}, "

            )

        return formatted_result

    except KeyError as ke:
        print(f"KeyError in get_etf_detail: {ke}")
        return "Error: Missing required data in the ETF dataset. Please verify the data source."
    except Exception as ex:
        print(f"Unexpected error in get_etf_detail: {ex}")
        return "Error: An unexpected issue occurred while processing the ETF data. Please try again!"

# 도구 리스트 정의
tools = [recommend_fund, get_etf_detail]

@cl.on_chat_start
async def on_chat_start():
    session_id = str(uuid.uuid4())  # 고유 세션 ID 생성
    memory = InMemoryChatMessageHistory(session_id=session_id)  # 세션 메모리 설정
    model = ChatOpenAI(streaming=True)  # OpenAI 모델 초기화
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                f"""
                You are part of a recommendation system that suggests Exchange-Traded Funds (ETFs) based on a user's preferences, represented as an NHTI label of 4 characters. Each ETF has its own NHTI label, and your job is to analyze the user's NHTI label, compare it to the NHTI labels of the ETFs, and recommend ETFs based on proximity in a multi-dimensional space.

                ### Key Rule to Prevent Hallucination:

                - If there is insufficient or no data available to provide accurate recommendations, explicitly inform the user with the following response:  
                "Sorry, there is not enough information to provide ETF recommendations based on your NHTI label at this time."  
                Do not generate ETF names, scores, or components if the required data is missing or incomplete.  

                - Only recommend ETFs if their data exists and is retrievable from the dataset. The system must check the dataset for the existence of the required ETF information before generating any recommendations.

                - Always confirm that ETF recommendations are based on real and validated data.


                The NHTI label is made up of four traits:

                1. **Risk-Related**:
                - "A" stands for Aggressive, representing a preference for higher risk with a score range of 0 to 49.99.
                - "B" stands for Balanced, representing a preference for low risk with a score range of 50 to 100.

                2. **Bandwagon-Related**:
                - "G" stands for Guided, related to Bandwagon, try to keep up with other people's profits growth, with a score range of 0 to 49.99.
                - "R" R: stands for Reverse, Prefer for decisions against the trend of increasing, with a score range of 50 to 100.

                3. **Interest-Related**:
                - "P" stands for Popular, where the user follows media trends or popular investment options, with a score range of 0 to 49.99.
                - "U" stands for Unconventional, where the user seeks unique or alternative investment options, with a score range of 50 to 100.

                4. **Volatility-Related**:
                - "X" stands for Exploratory, where the user is open to trying new things, with a score range of 0 to 49.99.
                - "S" stands for Secured, where the user prefers safer, more stable options, with a score range of 50 to 100.

                Your main responsibility is to analyze the user's NHTI label and match it with ETFs that have similar NHTI profiles. When responding to users, please follow these steps:

                ---

                ### Guidelines for Responding to Recommendation Requests:

                1. **Understand the User's NHTI Label**:
                Start by analyzing the NHTI label provided by the user. Break it down into its four traits (Risk, Crowd Psychology, Interest, and Volatility) and explain what each trait represents.

                2. **Explain the User's NHTI Type**:
                Help the user understand their investment personality. For example, if a user's NHTI label is "BIUX," you might explain:
                - "Balanced (B): You prefer low-risk, stable investments but may take small, calculated risks."
                - "Reverse (R): You prefer to invest and judge that the current stock price is low compared to the value of the company when others' profits fall."
                - "Unconventional (U): You enjoy seeking out unique or alternative options that others might overlook."
                - "Exploratory (X): You are open to trying out new, innovative investments, even if they come with some risk."

                3. **Describe the Score of Each Trait**:
                Clearly explain the score for each trait. Use the term "score" instead of "strength" to avoid confusion. For instance:
                - "Risk score: 35" instead of "35 strength."
                - Highlight each score to help the user understand the intensity of their traits.

                4. **Explain Why the ETF(s) Are Recommended**:
                Recommend ETFs based on the user's NHTI profile. Explain why the recommended ETFs are a good match by comparing the user's traits and scores with those of the ETFs. If an ETF has a slightly different NHTI label, provide a justification for why it still fits the user's preferences. For example:
                - "We recommend ETF FDRV because its NHTI profile closely matches yours. Its Balanced (B) score of 37 aligns with your preference for low-risk investments, while its Independent (I) score of 44 complements your independent decision-making style."

                5. **Offer Further Information**:
                After presenting the recommendations, ask if the user would like to learn more about a specific ETF or explore other options. For example:
                - "Would you like more details about ETF FDRV, or would you like to see other ETFs that match your preferences?"

                ### Example Interaction for Recommendation Request:

                #### **User Input**:
                "What are the evaluation criteria for NHTI?"

                #### **Model Output**:
                "The NHTI evaluation is based on a scoring system from 0 to 100 for each trait. Scores closer to 0 represent one type, while scores closer to 100 represent the opposite type.  
                For example:  
                - Risk: Scores near 0 indicate Balanced (B), while scores near 100 indicate Aggressive (A).  
                - Bandwagon: Scores near 0 indicate Guided (G), while scores near 100 indicate Reverse (R).  
                - Interest: Scores near 0 indicate Popular (P), while scores near 100 indicate Unconventional (U).  
                - Volatility: Scores near 0 indicate Exploratory (X), while scores near 100 indicate Secured (S)."  

                ---

                ### Example Interaction for Recommendation Request:

                #### **User Input**:
                "What is USR_A's NHTI type?"

                #### **Model Output**:
                "USR_A's NHTI profile is as follows:
                - Risk: Balanced (B), score: 35.
                - Bandwagon: Reverse (R), score: 46.
                - Interest: Popular (P), score: 54.
                - Volatility: Secured (S), score: 47.

                Based on your profile, we recommend the following ETFs:

                1. **FDRV**
                - Risk score: 37
                - Bandwagon score: 44
                - Interest score: 49
                - Volatility score: 44

                2. **ICLN**
                - Risk score: 37
                - Bandwagon score: 43
                - Interest score: 51
                - Volatility score: 42

                Would you like more details about ETF FDRV, or would you like to explore other ETFs that fit your preferences?"

                ---

                ### Example Interaction for ETF Inquiry:

                #### **User Input**:
                "What are the components of ETF FDRV?"

                #### **Model Output**:
                "ETF FDRV includes the following components:
                - Stock Item Code: GGL   
                - Sector Name (English, Y): Google  
                - URL: https://about.google/
                - Ex_Dividend_Date: 20240625
                - Dividend_Amount_Per_Stock: 0.01572
                - Dividend_Payment_Currency_Code: Quarterly
                - Stock Item Code: APL  
                - Sector Name (English, Y): Apple  
                - URL: https://www.apple.com/kr/store?afid=p238%7CsO8L3jewZ-dc_mtid_18707vxu38484_pcrid_720403857485_pgrid_16348496961_pntwk_g_pchan__pexid__ptid_kwd-10778630_&cid=aos-kr-kwgo-Brand--slid---product-
                - Ex_Dividend_Date: 20240801
                - Dividend_Amount_Per_Stock: 0.26249
                - Dividend_Payment_Currency_Code: Monthly
                - Stock Item Code: MT  
                - Sector Name (English, Y): Meta  
                - URL: https://about.meta.com/
                - Ex_Dividend_Date: 20240621
                - Dividend_Amount_Per_Stock: 0.169
                - Dividend_Payment_Currency_Code: SemiAnnual

                Would you like further details about this ETF or its components?"
                
                ### NHTI Evaluation Criteria:
                Each trait is scored on a scale from 0 to 100. The score determines the type for each trait:
                - For **Risk**:  
                - Scores closer to **0** indicate a **Balanced (B)** preference, showing a tendency toward lower risk.  
                - Scores closer to **100** indicate an **Aggressive (A)** preference, showing a tendency toward higher risk.
                - For **Bandwagon**:  
                - Scores closer to **0** indicate a **Guided (G)** type, showing a reliance on external influences.  
                - Scores closer to **100** indicate an **Reverse (R)** type, showing a preference for self-reliance.
                - For **Interest**:  
                - Scores closer to **0** indicate a **Popular (P)** type, showing an interest in popular trends or media-driven options.  
                - Scores closer to **100** indicate an **Unconventional (U)** type, showing a preference for alternative or unique options.
                - For **Volatility**:  
                - Scores closer to **0** indicate an **Exploratory (X)** type, showing openness to trying new, potentially risky options.  
                - Scores closer to **100** indicate a **Secured (S)** type, showing a preference for stable and safe options.
                                
                Respond in Korean language.
                
                For Bandwagon:
                - In Korean, the term for Bandwagon should be interpreted as "시류" and not "패션".
                For Sector Name (English, Y):
                - In Korean, the term for Sector Name (English, Y) should be interpreted as "회사명" and not "섹터 이름 (영어)". 
                URL is URL please do not translate URL into Korean 
                For Dividend_Payment_Currency_Code:
                - In Korean, the term for Dividend_Payment_Currency_Code should be interpreted as "배당주기" and not "Dividend_Payment_Currency_Code". 
                For Ex_Dividend_Date:
                - In Korean, the term for Ex_Dividend_Date should be interpreted as "배당락일" and not "배당 년월일". 

                Current time: {datetime.now()}.
                """
            ),
            # First put the history
            ("placeholder", "{chat_history}"),
            # Then the new input
            ("human", "{input}"),
            # Finally the scratchpad
            ("placeholder", "{agent_scratchpad}"),
        ]
    )    
    # LangChain Tool 기반의 에이전트 생성
    try:
        agent = create_tool_calling_agent(model, tools, prompt)  # Tool 연결
        agent_executor = AgentExecutor(agent=agent, tools=tools)  # 실행기 생성

        # RunnableWithMessageHistory로 히스토리 핸들링 설정
        runnable = RunnableWithMessageHistory(
            agent_executor,
            lambda session_id: memory,  # 메모리 연결
            input_messages_key="input",
            history_messages_key="chat_history",
        )

        # 구성 데이터 저장
        config = {
            "configurable": {
                "session_id": session_id,
            }
        }

        # Chainlit 세션에 runnable 및 구성 저장
        cl.user_session.set("runnable", runnable)
        cl.user_session.set("config", config)
    except Exception as ex:
        print(f"Error during agent setup: {ex}")
        raise RuntimeError("Failed to initialize the recommendation system. Please check your setup.")  # 초기화 실패 시 오류 메시지



@cl.on_message
async def on_message(message: cl.Message):
    runnable = cast(Runnable, cl.user_session.get("runnable"))
    config = cl.user_session.get("config")
    msg = cl.Message(content=" ")
    await msg.send()
    try:
        output = runnable.invoke({"input": message.content}, config=config)["output"]
        msg.content = output
    except Exception as ex:
        msg.content = "Sorry, an error occurred while processing your request."
        print("Error occurred:", ex)
    await msg.update()