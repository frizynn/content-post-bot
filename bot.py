import asyncio
from deep_research_py.deep_research import deep_research, write_final_report
from deep_research_py.ai.providers import get_ai_client
from deep_research_py.utils import set_service, set_model, get_model

async def generate_research_report(query: str):
    # Configure the service and model
    set_service("openai")  
    set_model("o3-mini")  # Or your preferred model

    # Get the AI client
    client = get_ai_client()
    
    # Research parameters
    breadth = 4  # Number of parallel searches
    depth = 2    # How deep to go in the research
    concurrency = 2  # Number of concurrent requests

    # Do the research
    research_results = await deep_research(
        query=query,
        breadth=breadth,
        depth=depth,
        concurrency=concurrency,
        client=client,
        model=get_model(),
    )

    # Generate the final report
    report = await write_final_report(
        prompt=query,
        learnings=research_results["learnings"],
        visited_urls=research_results["visited_urls"],
        client=client,
        model=get_model(),
    )

    return report, research_results["visited_urls"]

async def main():
    query = "Toma el rol de un experto en IA aplicada al marketing. Genera un resumen detallado de las noticias más relevantes sobre nuevas herramientas de inteligencia artificial aplicadas al marketing en las últimas 24 horas. Debe tener el formato de un daily newsletter. Objetivo: Proporcionar un newsletter conciso, valioso y accionable para profesionales del marketing interesados en IA.formato del boletín:**\n 1. **Título llamativo:** Crea un título atractivo y optimizado para SEO.\n 2. **Resumen ejecutivo:** Introducción de 2-3 frases sobre las tendencias generales del día.\n 3. **Noticias destacadas:**\n - 📰 **Titular de la noticia**\n - ✍️ **Fuente y enlace** (Asegurar que el enlace sea funcional y de una fuente confiable)\n - 📖 **Resumen de la noticia:** Explicación concisa y relevante sobre cómo esta innovación puede impactar el marketing digital.\n - 🚀 **Aplicaciones prácticas:** Descripción breve de cómo los profesionales del marketing pueden aprovechar la herramienta o tendencia.\n 4. **Análisis de impacto:** Un apartado final que explique en términos estratégicos cómo estas novedades pueden transformar el marketing digital en los próximos meses"
    report, sources = await generate_research_report(query)
    
    # Save the report
    with open("dog_report.md", "w", encoding="utf-8") as f:
        f.write("# Dog Research Report\n\n")
        f.write(report)
        f.write("\n\n## Sources\n")
        for url in sources:
            f.write(f"- {url}\n")

    print("Research report has been generated and saved to dog_report.md")

if __name__ == "__main__":
    asyncio.run(main())