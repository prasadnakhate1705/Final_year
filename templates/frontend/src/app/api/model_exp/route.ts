import  {NextResponse} from "next/server";
import {generateModelText} from "../../lib/openai";

export const runtime = "edge";

export async function POST(req: Request){
    const body = await req.json();
    const model_text = body.model_text;
    console.log(model_text);
    // trigger the generateModelText fxn
    const modelExplaination = await generateModelText(model_text);
    if(!modelExplaination) {
        return new NextResponse('failed to generate model_explainations', {status: 500});
    }


    return new NextResponse(modelExplaination);
}