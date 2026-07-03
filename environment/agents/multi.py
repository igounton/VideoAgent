import ast
import json
import logging
import re

import regex
import yaml
from typing import Dict, List
from pathlib import Path
from environment.agents.base import FunctionRegistry
from environment.config.llm import claude
from environment.roles import get_agent_class


MAX_Retries = 3
with open('environment/config/graph.txt', 'r', encoding='utf-8') as f:
    graph_prompt = f.read()

class MultiAgent:
    def __init__(self):
        FunctionRegistry.auto_register("environment/roles")
        self.registry = FunctionRegistry.get_registry()
        self.intents_path = 'environment/config/intents.yml'
        self.intents = self.load_yml(self.intents_path)
        self.current_requirement = None

    def load_yml(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def get_tools_by_intents(self, intent_list):
        if not intent_list:
            return None

        tool_names = []
        for intent in intent_list:
            if intent in self.intents:
                tool_names.extend(self.intents[intent])

        registry_dict = json.loads(self.registry)
        matched_tools = {}

        for tool_name in tool_names:
            if tool_name in registry_dict:
                matched_tools[tool_name] = registry_dict[tool_name]
            else:
                print(f"Warning: Tool '{tool_name}' not found in registry")
                return None

        return matched_tools

    def intents_analysis(self, reqs, previous_intents=None, reflection=None):
        intents = list(self.intents.keys())

        if previous_intents is None:
            intents_prompt = f"""
You are an intent analyst.  
I will provide a set of candidate intents and a user's requirements.
Please select as many relevant candidate intents as possible that match the user's requirements. Consider factors such as keywords, audio types (speech, song, music, etc.), and other relevant dimensions.  

Candidate intents:  
{intents}  

User requirements:  
{reqs}  

Please Only output pure List format:
['intent1', 'intent2', ...]

Note! Don't output any analysis and explanations!
            """
        else:
            intents_prompt = f"""
You are an intent analyst.  
Previous analysis attempt failed with the following reflection:
{reflection}

Previous selected intents:
{previous_intents}

Please re-analyze the user's requirements with the candidate intents below, considering the reflection.
Select as many relevant candidate intents as possible.

Candidate intents:  
{intents}  

User requirements:  
{reqs}  

Please Only output pure List format:
['intent1', 'intent2', ...]

Note! Don't output any analysis and explanations!
            """

        try:
            response = claude(user=intents_prompt)
            match = re.search(r'\[.*\]', response.choices[0].message.content.strip())
            if match:
                intents_list = match.group(0)
                print("Filtered intents_list：\n", intents_list)
                # Use literal_eval instead of eval: the string comes from LLM
                # output, so eval() would execute arbitrary Python code.
                return ast.literal_eval(intents_list)
            else:
                raise ValueError("No valid intent list found in response")
        except Exception as e:
            print(f"Error parsing intents: {e}")
            return None

    def generate_agent_graph(self, requirement, tools, pre_agent_graph=None, pre_agent_chain=None, pre_user_input_graph=None, reflection=None):
        """Generate agent chain for a requirement"""
        if reflection is None:
            user_prompt = f"""
{graph_prompt}

User requirements:
{requirement}

Metadata of registered agents:
{tools}
        """
        else:
            user_prompt = f"""
Previous Prompt Used to Generate the Agent System:
{graph_prompt}

User requirements:
{requirement}

Metadata of registered agents:
{tools}

Previously generated Agent Graph:
{pre_agent_graph}

Previously generated Agent Chain:
{pre_agent_chain}

Previously required User Input Graph
{pre_user_input_graph}

Reflection on the previous design:
{reflection}

Refine the previously designed Agent System (Agent Graph, Agent Chain, and User Input Graph) based on reflection insights to address identified issues.
Output Format: Maintain the original prompt structure used for system generation.
        """

        response = claude(user=user_prompt)
        raw_chain = response.choices[0].message.content.strip()
        match = regex.search(r'\{(?:[^{}]|(?R))*\}', raw_chain, regex.DOTALL)
        try:
            parsed = json.loads(match.group())
            print(parsed)
            if not all(key in parsed for key in
                       ("Feasibility", "Agent Graph", "Agent Chain", "User Input Graph", "Reasoning")):
                raise ValueError("Missing required fields")
            return parsed
        except (json.JSONDecodeError, ValueError) as e:
            logging.error(f"JSON validation failed: {e}")
            return None

    def judge_agent_graph(self, agent_graph, agent_chain, user_inputs, reqs, tools):
        try:
            judge_prompt = f"""
You are an agent graph validation system.
I will provide:
1. User Requirement
2. Registered agent metadata
3. Candidate Agent Graph
4. An Agent Chain derived from the Candidate Agent Graph
5. Required User Inputs

User Requirements:
{reqs}

Metadata of registered agents:
{tools}

Task: Evaluate the candidate agent graph:

Candidate Agent Graph:
{agent_graph}

Agent Chain:
{agent_chain}

Required User Inputs:
{user_inputs}

Evaluation Criteria:
1. Based on the Metadata of registered agents and parameter passing in the Agent Graph, determine from multiple aspects whether the user requirements can be fulfilled:
   - Execution sequence of agents in the Agent Graph
   - For parameter nodes with no incoming edges, they are uniformly considered as user inputs, but it is necessary to determine whether they should be provided by the user or by the parent agent
   - Validate that the necessary output parameters are correctly routed to the intended agent and the expected input parameters.
2. There should be no functionally redundant agents (e.g., repeatedly adding audio tracks to a video).
3. For vaguely mentioned requirements in user needs, lenient evaluation is acceptable. For example, if the user requests audio quality improvement, it's sufficient as long as at least one relevant agent in the graph meets this requirement.

Please Only output pure JSON format:
{{
"Result": '0' if correct else '1',
"Reasoning": Concisely state the key reasons why a score of '0' or '1' was assigned (<100 words).
}}
"""
            response = claude(user=judge_prompt)
            judge_res = response.choices[0].message.content.strip()
            reflection_prompt = f"""
            You are an agent graph reflection system.
            I will provide:
            1. User Requirement
            2. Registered agent metadata
            3. Candidate Agent Graph
            4. An Agent Chain and User Input Graph derived from the Candidate Agent Graph
            5. Previous validation result 

            User Requirements:
            {reqs}

            Metadata of registered agents:
            {tools}

            Task: Evaluate the candidate agent graph:

            Candidate Agent Graph:
            {agent_graph}

            Agent Chain:
            {agent_chain}

            Required User Input Graph:
            {user_inputs}

            Previous validation result:
            {judge_res}

            Reflection Task:
            1. If the previous validation result is '0', please reflect on whether there were any overlooked aspects based on the **Evaluation Criteria** and the reasoning behind the previous validation result.
            2. If the previous validation result is '1', please reflect on whether the reasoning behind the previous validation result was correct.

            Evaluation Criteria:
            1. Based on the Metadata of registered agents and parameter passing in the Agent Graph, determine from multiple aspects whether the user requirements can be fulfilled:
               - Execution sequence of agents in the Agent Graph
               - For parameter nodes with no incoming edges, they are uniformly considered as user inputs, but it is necessary to determine whether they should be provided by the user or by the previous agent
               - Validate that the necessary output parameters are correctly routed to the intended agent and the expected input parameters.
               - Validate that the output parameters' description and type match the input requirements of the next agent.
               - Not all output parameters are necessarily mapped to the input requirements of the next agent. Redundant output parameters may exist, but they should not interfere with the fulfillment of user requirements.
            2. There should be no functionally redundant agents (e.g., repeatedly adding audio tracks to a video).
            3. For vaguely mentioned requirements in user needs, lenient evaluation is acceptable. For example, if the user requests audio quality improvement, it's sufficient as long as at least one relevant agent in the graph meets this requirement.

            Please Only output pure JSON format:
            {{
            "Result": '0' if correct else '1',
            "Reasoning": Concisely state the key reasons why a score of '0' or '1' was assigned (<100 words).
            }}
            """
            response = claude(user=reflection_prompt)
            reflection_res = response.choices[0].message.content.strip()
            json_match = re.search(r'\{.*\}', reflection_res, re.DOTALL)
            if json_match:
                print(json_match.group())
                return json_match.group()
            else:
                logging.error("No JSON found in the response")
                return None

        except Exception as e:
            logging.error(f"Judge error: {e}")
            return None

    def execute_agent_chain(self, agent_graph, agent_chain, user_input_graph):
        # 1. 初始化上下文
        context = {}

        # 2. 处理用户输入
        user_inputs = {}
        for user_input_node in user_input_graph:
            # 获取用户输入值（这里需要实际的前端交互，此处用模拟数据）
            value = input(f"Please Input {user_input_node['node']} ({user_input_node['description']}):  ")

            # 将用户输入映射到各个Agent的输入参数
            for link in user_input_node['links']:
                for agent_name, input_param in link.items():
                    context_key = f"{agent_name}.{input_param}"
                    context[context_key] = value

        # 3. 按顺序执行Agent链
        for agent_name in agent_chain:
            # 获取Agent配置信息
            agent_config = next(agent for agent in agent_graph if agent["node"] == agent_name)

            # 4. 准备输入参数
            inputs = {}
            for input_param in agent_config["inputs"]:
                context_key = f"{agent_name}.{input_param['name']}"

                if context_key not in context:
                    raise ValueError(f"Missing required parameter: {context_key}")

                inputs[input_param["name"]] = context[context_key]

            # 5. 实例化并执行Agent
            agent_class = get_agent_class(agent_name)
            agent_instance = agent_class()
            result = agent_instance.execute(**inputs)

            # 6. 处理输出参数
            for output in agent_config["outputs"]:
                output_value = result.get(output["name"])

                # 将输出值连接到下游Agent的输入
                for link in output["links"]:
                    for target_agent, target_input in link.items():
                        context_key = f"{target_agent}.{target_input}"
                        context[context_key] = output_value

        return context

    def process_requirement(self, requirement: str):

        max_reflections = 3
        reflection_attempts = 0
        previous_intents = None
        intent_reflection = None
        graph_reflection = None
        agent_chain = None
        tools = None
        current_intents = None
        previous_agent_graph = None
        previous_agent_chain = None
        previous_user_input_graph = None

        while reflection_attempts < max_reflections:
            # Step 1: Intent Analysis with retries (only if we don't have current_intents)
            if current_intents is None:
                for attempt in range(MAX_Retries):
                    try:
                        intent_list = self.intents_analysis(
                            requirement,
                            previous_intents,
                            intent_reflection
                        )
                        if intent_list:
                            tools = self.get_tools_by_intents(intent_list)
                            if tools:
                                print(f"Tools obtained successfully: {list(tools.keys())}")
                                current_intents = intent_list
                                break
                            else:
                                print(f"No matching tools found")
                                continue
                        else:
                            print(f"Intent analysis returned empty list")
                    except Exception as e:
                        logging.warning(f"Intent analysis attempt {attempt + 1} failed: {e}")
                        if attempt == MAX_Retries - 1:
                            logging.error(f"Reached max retries for intent analysis")
                            return 1
            else:
                # Reuse existing intents and tools
                intent_list = current_intents
                tools = self.get_tools_by_intents(intent_list)

            # Step 2: Generate agent graph with retries
            for attempt in range(MAX_Retries):
                try:
                    print(f"Trying to generate agent graph (attempt {attempt + 1}/{MAX_Retries})")
                    agent_data = self.generate_agent_graph(
                        requirement,
                        tools,
                        previous_agent_graph,
                        previous_agent_chain,
                        previous_user_input_graph,
                        graph_reflection
                    )
                    if not agent_data:
                        print(f"Agent graph generation returned empty value")
                        if attempt == MAX_Retries - 1:
                            logging.error(f"Reached max retries for agent graph generation")
                            return 1
                        continue
                    previous_agent_graph = agent_data["Agent Graph"]
                    previous_agent_chain = agent_data["Agent Chain"]
                    previous_user_input_graph = agent_data["User Input Graph"]
                    print(f"Agent graph generated successfully")
                    try:
                        agent_chain = agent_data["Agent Chain"]
                        print("First possible execution order:", agent_chain)
                    except ValueError as e:
                        print("Error:", e)
                    break
                except Exception as e:
                    logging.warning(f"Agent graph generation attempt {attempt + 1} failed: {e}")
                    if attempt == MAX_Retries - 1:
                        logging.error(f"Reached max retries for agent graph generation")
                        return 1

            print(f"Agent graph parsed successfully. Feasibility: {agent_data['Feasibility']}")

            if agent_data["Feasibility"] == "Infeasible":
                # If graph is infeasible, reflect on intents
                intent_reflection = agent_data["Reasoning"]
                graph_reflection = None
                current_intents = None  # Reset to trigger new intent analysis
                reflection_attempts += 1
                print(
                    f"Graph marked as infeasible, reflecting on intents (attempt {reflection_attempts}/{max_reflections})")
                continue

            # Step 3: Perform judgment (only if feasible)
            try:
                print(f"Starting agent graph judgment")
                for attempt in range(MAX_Retries):
                    try:
                        judge_response = self.judge_agent_graph(
                            agent_data["Agent Graph"],
                            agent_chain,
                            agent_data["User Input Graph"],
                            requirement,
                            tools
                        )
                        if judge_response:
                            judge_data = json.loads(judge_response)
                            judge_result = {
                                'Result': judge_data.get("Result", '1'),
                                'Reasoning': judge_data.get("Reasoning", 'No reasoning provided')
                            }
                            print(f"Judgment result: {'Pass' if judge_result['Result'] == '0' else 'Fail'}")

                            if judge_result['Result'] == '0':
                                return agent_data

                            # If judgment fails, reflect on graph generation only
                            graph_reflection = judge_result['Reasoning']
                            intent_reflection = None
                            reflection_attempts += 1
                            print(
                                f"Judgment failed, reflecting on graph (attempt {reflection_attempts}/{max_reflections})")
                            break
                    except Exception as e:
                        logging.warning(f"Judgment attempt {attempt + 1} failed: {e}")

            except Exception as e:
                logging.error(f"Judgment process failed: {e}")
                return 1

        # If we get here, all reflection attempts failed or max reflections reached
        return 1

    def run(self):
        requirement = input("User Requirement:")
        result = self.process_requirement(requirement)
        # process_requirement returns an int (1) instead of a dict when it
        # fails to converge on a valid plan. Guard against that here so the
        # program exits cleanly instead of raising a TypeError on lookup.
        if not isinstance(result, dict):
            print("Failed to generate a valid agent plan after multiple attempts. Exiting.")
            return
        agent_graph = result["Agent Graph"]
        agent_chain = result["Agent Chain"]
        user_input_graph = result["User Input Graph"]
        context = self.execute_agent_chain(agent_graph, agent_chain, user_input_graph)
        print(context)
